# Versper-V1-Evo: An Autonomous Agentic Model that Operates in Iterative Execution Loops.

<div align="center">
  <img src="assets/VersperAI_Banner.png" alt="Versper-V1-Evo">
</div>

## 1. Quick start

### Init Train Environment

```bash
uv venv && uv sync && source .venv/bin/activate
```

### Fetch Data: Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k

> Q1: Why use recursisve code & scientific think long data, not just ordinary chat short data?  
> A1: just code & scientific have longest tokens, biggest value density, closest relational connection

> Q2: How to set data shema to let model evolution to what we need feature  
> A2: i want model can self-evolution in iterative loop can task-oriented learning so use ORPO + Process Reward Modeling and use prompt + chosen + rejected as data schema 

```bash
curl -LsSf https://hf.co/cli/install.sh | bash && hf auth login && mkdir -p data && hf download versperai/Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k --repo-type dataset --local-dir . && cd ..
```

### Pull Model: Versper-V1-Instruct  

```bash
mkdir -p model && cd model && hf download versperai/Versper-V1-Instruct --local-dir . && cd ..
```

## 2. EDA - data/eda.py

### Choose max_seq_length can cover all data sequence lengths and %64

```python
# analysis data distribution and data features
# used in train make sure no data overflow

max_seq_length = 4480 #  max_prompt_length=1280 + max_completion_length=3264 
overflow = 0

for p, c, r in zip(prompt_lens, chosen_lens, rejected_lens):
    # ORPO：prompt + chosen / rejected
    if p + c > max_seq_length or p + r > max_seq_length:
        overflow += 1
```

### Random Sample to make sure the data is complete and correct loaded 

```python
for _ in range(2):
    ex = random.choice(data)

    print("PROMPT:\n", ex["prompt"][:300])
    print("\nCHOSEN:\n", ex["chosen"][:300])
    print("\nREJECTED:\n", ex["rejected"][:300])
    print("=" * 80)
```

## 3. Eval - eval/inference.py

```python
# Construct Conversation
messages = [{"role": "user", "content": "你好."}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

## 4. Train - trainer/train_orpo.py

> Watch the orpo-evo post-trian logs in [Training  Metric Logging](https://swanlab.cn/@HaibaraYuki/Versper-V1-ORPO/runs)  
> scientific exploration : pre-train : post-train = 3 : 1 : 1

> batch_size	4, grad_accum 3 => total batch 12 trained in two 4090 48G VAGM

### 4.1 Steps - 125

<div align="center">
  <img src="assets/train_orpo_log/125steps.png" alt="125steps">
</div>

> **Core Loss:** Steady decline in <code>nll_loss</code> and stable <code>grad_norm</code> (<0.5) indicate robust language modeling and stable gradients.  
> **ORPO Metrics:** <code>margins</code> and <code>log_odds_ratio</code> are rising, proving the model is effectively widening the gap between <code>chosen</code> and <code>rejected</code> responses.  
> **Reward Accuracy:** Sustained at 1.0, showing strong discriminative power on the Vesper-V1 dataset.  
> **Log Probabilities:** Asymmetric growth (chosen > rejected) confirms the model prioritizes high-quality outputs over mere imitation.  
> **Conclusion:** Training is optimal. Convergence is expected around step 1500.  

### 4.2 
