from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig
import swanlab
import os

# =========================================================
# 🧠 0. SwanLab 初始化（必须在 trainer 前）
# =========================================================
swanlab.init(
    project="Versper-V1-ORPO",
    experiment_name="run-3125",
)

# =========================================================
# 1️⃣ Load Model (4bit + Unsloth)
# =========================================================
model, processor = FastLanguageModel.from_pretrained(
    model_name="/root/yijia-tmp/model",
    max_seq_length=4480,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Use text-only tokenizer (processor is a Qwen3VLProcessor which treats first positional arg as image)
text_tokenizer = processor.tokenizer
text_tokenizer.padding_side = "right"
text_tokenizer.fix_mistral_regex = True

# =========================================================
# 2️⃣ LoRA (Unsloth PEFT)
# =========================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =========================================================
# 3️⃣ Dataset (LOCAL JSONL, NO TRUNCATION)
# =========================================================
dataset = load_dataset(
    "json", data_files="/root/yijia-tmp/data/orpo_train.jsonl", split="train"
)

# =========================================================
# 4️⃣ ORPO Trainer
# =========================================================
trainer = ORPOTrainer(
    model=model,
    tokenizer=text_tokenizer,
    processing_class=text_tokenizer,
    train_dataset=dataset,
    args=ORPOConfig(
        # =========================
        # 🧠 sequence config
        # =========================
        max_length=4480,
        max_prompt_length=1280,
        max_completion_length=3264,
        # =========================
        # 💾 memory config (48GB safe)
        # =========================
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        # =========================
        # ⚖️ ORPO
        # =========================
        beta=0.1,
        # =========================
        # 🚀 optimizer
        # =========================
        optim="adamw_8bit",
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        # =========================
        # 📊 logging
        # =========================
        logging_steps=5,
        report_to="swanlab",
        # =========================
        # 💾 saving
        # =========================
        save_steps=200,
        save_total_limit=2,
        # =========================
        # 🚀 training scale
        # =========================
        max_steps=3125,  # ≈ 1 epoch (25k dataset)
        bf16=True,
        output_dir="./orpo_output",
    ),
)

# =========================================================
# 5️⃣ Train
# =========================================================
trainer.train()

# =========================================================
# 6️⃣ Save LoRA adapter
# =========================================================
model.save_pretrained("./orpo_output/lora")
processor.save_pretrained("./orpo_output/lora")

# =========================================================
# 7️⃣ Merge full model for inference
# =========================================================
model = model.merge_and_unload()
model.save_pretrained("./orpo_output/merged")
processor.save_pretrained("./orpo_output/merged")

print("✅ ORPO training finished successfully!")
