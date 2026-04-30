# Versper-V1-Evo: An Autonomous Agentic Model that Operates in Iterative Execution Loops.

<div align="center">
  <img src="assets/VersperAI_Banner.png" alt="Versper-V1-Evo">
</div>

## 1. Quick start

### Init Train Environment

```bash
uv venv && uv sync
```

### Fetch Data: Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k

```bash
curl -LsSf https://hf.co/cli/install.sh | bash && hf auth login && mkdir -p data && hf download versperai/Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k --repo-type dataset --local-dir . && cd ..
```

### Pull Model: Versper-V1-Instruct  

```bash
mkdir -p model && cd model && hf download versperai/Versper-V1-Instruct --local-dir . && cd ..
```

## 2. EDA

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


