import json
import numpy as np
import random

DATA_PATH = "/root/yijia-tmp/data/orpo_train.jsonl"

print("📦 Loading local dataset:", DATA_PATH)

data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print("✅ Total samples:", len(data))


# =========================
# 1️⃣ 字段检查
# =========================
required_fields = ["prompt", "chosen", "rejected"]

for i, ex in enumerate(data):
    for f in required_fields:
        if f not in ex:
            raise ValueError(f"❌ Missing field {f} at index {i}")

print("✅ All required fields exist")


# =========================
# 2️⃣ 长度统计（字符级）
# =========================
def l(x):
    return len(str(x))


prompt_lens, chosen_lens, rejected_lens = [], [], []

bad = []

for i, ex in enumerate(data):
    p = l(ex["prompt"])
    c = l(ex["chosen"])
    r = l(ex["rejected"])

    prompt_lens.append(p)
    chosen_lens.append(c)
    rejected_lens.append(r)

    if p == 0 or c == 0 or r == 0:
        bad.append((i, "empty field"))

    if p > 12000 or c > 12000 or r > 12000:
        bad.append((i, "too long"))

print("\n📊 dataset overview:")
print("prompt avg:", np.mean(prompt_lens))
print("chosen avg:", np.mean(chosen_lens))
print("rejected avg:", np.mean(rejected_lens))


# =========================
# 3️⃣ ORPO 长度风险（关键）
# =========================
max_seq_length = 4480  # 你训练时用的

overflow = 0

for p, c, r in zip(prompt_lens, chosen_lens, rejected_lens):
    # ORPO 本质：prompt + chosen / rejected
    if p + c > max_seq_length or p + r > max_seq_length:
        overflow += 1

print("\n⚠️ ORPO overflow check:")
print("overflow samples:", overflow)
print("ratio:", overflow / len(data))


# =========================
# 4️⃣ 异常样本
# =========================
print("\n🚨 bad samples:", len(bad))

for i, reason in bad[:10]:
    print(f"\n[{i}] {reason}")
    print(data[i])
    print("-" * 60)


# =========================
# 5️⃣ 随机看样本
# =========================
print("\n🧪 random samples:\n")

for _ in range(2):
    ex = random.choice(data)

    print("PROMPT:\n", ex["prompt"][:300])
    print("\nCHOSEN:\n", ex["chosen"][:300])
    print("\nREJECTED:\n", ex["rejected"][:300])
    print("=" * 80)

"""
(base) root@225b37b720f9:~# python check_data.py
📦 Loading local dataset: /root/yijia-tmp/data/orpo_train.jsonl
✅ Total samples: 25000
✅ All required fields exist

📊 dataset overview:
prompt avg: 1069.4846
chosen avg: 2869.35972
rejected avg: 1057.9446

⚠️ ORPO overflow check:
overflow samples: 2539
ratio: 0.10156

🚨 bad samples: 0

🧪 random samples:

PROMPT:
 Instruction: You are Versper-V1-Evo, an autonomous agent that operates in iterative execution loops. Perform a rigorous, fact-based self-assessment of the model "Gemma-2-9B-it" on the metric "DS-1000" (current estimated score: 84%).

Your task:
1. Break down the model's likely strengths and weaknes

CHOSEN:
 **Step 1: Rigorous Self-Assessment (Fact-Based)**

Current estimated performance on DS-1000: 84%

**Strengths (based on architecture and training data patterns):**
- Strong performance on structured reasoning tasks due to post-training alignment (similar to Llama-3.3 and Qwen2.5 results).
- Good too

REJECTED:
 Current estimated performance on DS-1000: 84%

**Strengths (based on architecture and training data patterns):**
- Strong performance on structured reasoning tasks due to post-training alignment (similar to Llama-3.3.
- Good tool-calling when prompts are explicit (ReAct-style).
- Solid long-context
================================================================================
PROMPT:
 Instruction: You are Versper-V1-Evo, an autonomous agent that operates in iterative execution loops. Your current system prompt for Claude-3.5-Sonnet is producing suboptimal results on "expand knowledge cutoff while preserving alignment".

Task:
1. Critically analyze the current prompt (I will provi

CHOSEN:
 **Recursive Prompt Optimization Loop (Using ReAct + Reflexion)**

**Current Prompt Diagnosis (Fact-Based Critique):**
- Too generic → no domain-specific scaffolding (violates "specificity" principle from Prompt Engineering Guide).
- No explicit self-critique trigger → models skip reflection (see Ref

REJECTED:
 **Version 2 (Add ReAct + basic reflection Scaffolding)**
[Full detailed prompt incorporating standard structured prompting branching + backtracking logic + explicit "explore 3 paths, prune weakest" instruction. Adds ~180 tokens but improves GPQA by 11-14 points in internal tests.]
================================================================================
"""
