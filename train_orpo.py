from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig

# =====================================================
# 1️⃣ Load Model (4bit + BF16)
# =====================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/yijia-tmp/model",
    max_seq_length=4480,  # ⭐ full context
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# tokenizer patch (Mistral regex warning fix)
tokenizer.fix_mistral_regex = True

# ⚠️ training mode (not inference)
FastLanguageModel.for_training(model)

# =====================================================
# 2️⃣ LoRA (48GB optimized)
# =====================================================
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

# =====================================================
# 3️⃣ Load Local Dataset
# =====================================================
dataset = load_dataset(
    "json", data_files="/root/yijia-tmp/data/orpo_train.jsonl", split="train"
)


# 不做截断（你已经分析过长度分布）
def preprocess(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


dataset = dataset.map(preprocess)

# =====================================================
# 4️⃣ ORPO Config (48GB 4090 full config)
# =====================================================
trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=ORPOConfig(
        # =========================
        # 🧠 context config (FULL 4480)
        # =========================
        max_length=4480,
        max_prompt_length=1280,
        max_completion_length=3264,
        # =========================
        # 💾 memory config
        # =========================
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        # =========================
        # ⚖️ ORPO core
        # =========================
        beta=0.1,
        # =========================
        # 🚀 optimizer
        # =========================
        optim="adamw_8bit",
        # =========================
        # 📉 learning
        # =========================
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        # =========================
        # 🧾 logging (SwanLab)
        # =========================
        logging_steps=5,
        report_to="swanlab",
        swanlab_project="Versper-V1-ORPO",
        swanlab_entity="HaibaraYuki",
        # =========================
        # 💾 saving
        # =========================
        save_steps=200,
        save_total_limit=2,
        # =========================
        # 🚀 training scale
        # =========================
        max_steps=3125,
        bf16=True,
        output_dir="./orpo_output",
    ),
)

# =====================================================
# 5️⃣ Train
# =====================================================
trainer.train()

# =====================================================
# 6️⃣ Save (LoRA + merged model)
# =====================================================
model.save_pretrained("./orpo_output/lora")
tokenizer.save_pretrained("./orpo_output/lora")

model = model.merge_and_unload()
model.save_pretrained("./orpo_output/merged")
tokenizer.save_pretrained("./orpo_output/merged")

print("✅ ORPO training finished successfully!")
