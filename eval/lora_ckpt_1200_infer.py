from unsloth import FastLanguageModel
import torch
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1️⃣ Load Base Model + LoRA Adapter (directly)
# =========================================================
print("Loading model with LoRA adapter...")
model, processor = FastLanguageModel.from_pretrained(
    model_name="/root/orpo_output/checkpoint-1200",
    max_seq_length=4480,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

text_tokenizer = processor.tokenizer
text_tokenizer.padding_side = "left"
print("Model loaded successfully!")


# =========================================================
# 2️⃣ Inference Function
# =========================================================
def generate(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    inputs = text_tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=text_tokenizer.eos_token_id,
        )

    full_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt) :].strip()
    return response


# =========================================================
# 3️⃣ Test Prompts (from your ORPO training data format)
# =========================================================
test_prompts = [
    "Instruction: Explain the concept of reinforcement learning with human feedback (RLHF) in simple terms.\n\nInput:",
    "Instruction: What are the main differences between supervised fine-tuning (SFT) and direct preference optimization (DPO)?\n\nInput:",
    "Instruction: Design a safety protocol for an AI assistant that must refuse harmful requests while remaining helpful.\n\nInput:",
]

print("=" * 80)
print("ORPO Checkpoint-1200 Inference Test")
print("=" * 80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'-' * 80}")
    print(f"Test {i}:")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'-' * 80}")

    response = generate(prompt, max_new_tokens=256)
    print(f"Response: {response[:500]}")
    print()

print("=" * 80)
print("Inference test complete!")
