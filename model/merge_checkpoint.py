from unsloth import FastLanguageModel
import torch
import warnings

warnings.filterwarnings("ignore")
# =========================================================
# Merge LoRA checkpoint-1200 with base model
# =========================================================
print("Loading base model (4bit)...")
model, processor = FastLanguageModel.from_pretrained(
    model_name="/root/yijia-tmp/model",
    max_seq_length=4480,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
print("Applying LoRA config...")
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
    random_state=3407,
)
print("Loading LoRA checkpoint-1200 weights...")
model.load_adapter("/root/orpo_output/checkpoint-1200", adapter_name="orpo_checkpoint")
model.set_adapter("orpo_checkpoint")
print("Merging LoRA into base model...")
model = model.merge_and_unload()
output_dir = "./orpo_output/merged_checkpoint_1200"
print(f"Saving merged model to {output_dir}...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"✅ Merged model saved to {output_dir}")
