from unsloth import FastLanguageModel
import torch
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# Merge LoRA checkpoint-1200 with base model and export
# =========================================================
print("Loading base model + LoRA checkpoint-1200 directly...")
model, processor = FastLanguageModel.from_pretrained(
    model_name="/root/orpo_output/checkpoint-1200",
    max_seq_length=4480,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Method 1: Save as GGUF (recommended for 4bit)
print("\nSaving as GGUF...")
model.save_pretrained_gguf(
    "./orpo_output/merged_checkpoint_1200_gguf",
    processor,
    quantization_method="q4_k_m",
)
print("✅ GGUF model saved to ./orpo_output/merged_checkpoint_1200_gguf")
