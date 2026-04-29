from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig

# ==============================================
# 1️⃣ 加载模型
# ==============================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/yijia-tmp/model",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
tokenizer.fix_mistral_regex = True
FastLanguageModel.for_inference(model)

# ==============================================
# 2️⃣ LoRA
# ==============================================
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

# ==============================================
# 3️⃣ Dataset
# ==============================================
dataset = load_dataset(
    "versperai/Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k", split="train"
)


def preprocess(example):
    # 自动裁切防显存炸
    return {
        "prompt": example["prompt"][:2000],
        "chosen": example["chosen"][:2000],
        "rejected": example["rejected"][:2000],
    }


dataset = dataset.map(preprocess)

# ==============================================
# 4️⃣ ORPO Trainer 配置
# ==============================================
trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=ORPOConfig(
        max_length=2048,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        beta=0.1,
        optim="adamw_8bit",
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        logging_steps=5,
        report_to="swanlab",  # ⭐ 这里启用 SwanLab
        swanlab_project="Versper-V1-ORPO",  # SwanLab项目名
        swanlab_entity="HaibaraYuki",  # SwanLab账户名
        save_steps=200,
        save_total_limit=2,
        max_steps=3000,
        bf16=True,
        output_dir="./orpo_output",
    ),
)

# ==============================================
# 5️⃣ 训练
# ==============================================
trainer.train()

# ==============================================
# 6️⃣ 保存 LoRA + Merge
# ==============================================
model.save_pretrained("./orpo_output/lora")
tokenizer.save_pretrained("./orpo_output/lora")

model = model.merge_and_unload()
model.save_pretrained("./orpo_output/merged")
tokenizer.save_pretrained("./orpo_output/merged")
