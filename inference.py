from unsloth import FastLanguageModel
import torch

# 1. 加载模型和处理器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/checkpoint_4bit",  # yijia-tmp/model
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)

# 修复 tokenizer 潜在的正则问题
tokenizer.fix_mistral_regex = True

# 开启推理模式优化
FastLanguageModel.for_inference(model)

# 2. 构造对话
# 注意：对于这类混合模型，简单的字符串 content 往往比嵌套的 list 更稳健
messages = [{"role": "user", "content": "你好."}]

# 生成 Prompt
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 3. 准备输入 (使用 processor 逻辑显式排除图片)
inputs = tokenizer(
    text=[prompt],
    images=None,
    return_tensors="pt",
).to("cuda")

# 4. 执行推理
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,  # 降低随机性，防止跑题
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# 5. 【关键修复】只截取模型新生成的 ID
# 之前的报错可能是因为 decode 了整个 sequence 包含了 Prompt 干扰
input_len = inputs.input_ids.shape[1]
generated_ids = outputs[0][input_len:]

response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("=== MODEL OUTPUT ===")
print(response.strip())


"""
=== MODEL OUTPUT ===
1.  **Analyze the User's Input:**
    *   Input: "你好." (Hello.)
    *   Language: Chinese
    *   Intent: Greeting / Starting a conversation.

2.  **Determine the Appropriate Response:**
    *   Acknowledge the greeting warmly.
    *   Respond in the same language (Chinese).
    *   Offer assistance.

3.  **Drafting the Response (Internal Monologue/Trial):**
    *   *Option 1:* 你好！有什么我可以帮你的吗？ (Hello! Is there anything I can help you with?) - *Standard, polite.*
    *   *Option 2:* 你好！很高兴见到你。今天有什么我可以为你做的吗？ (Hello! Nice to see you. Is there anything I can do for you today?) - *A bit more enthusiastic.*
    *   *Option 3:* 你好！请问有什么我可以帮您的？ (Hello! How can I help you?) - *Slightly more formal.*

4.  **Selecting the Best Response:**
    *   Option 1 is a great balance of friendly and helpful. Let's go with a variation of it.

5.  **Final Polish (Chinese):**
    *   "你好！很高兴和你聊天。请问有什么我可以帮你的吗？" (Hello! Nice to chat with you. Is there anything I can help you with?)

6.  **Output Generation:** (Translate the polished thought into the final output).
</think>

你好！很高兴和你聊天。请问有什么我可以帮你的吗？
"""

"""
=== MODEL OUTPUT ===
1.  **Deconstruct the Request:**
    *   **Topic:** ORPO (Omnipotent Retrieval-Augmented Generation with Preference Optimization).
    *   **Goal:** Explain it in *simple terms*.
    *   **Target Audience:** Someone who might not be a deep learning expert but wants to understand the core concept and why it matters.

2.  **Understand ORPO (The Technical Core):**
    *   *What is it?* A method for training Large Language Models (LLMs) to be better at Retrieval-Augmented Generation (RAG).
    *   *What is RAG?* Giving an LLM access to external documents (like a search engine) so it can answer questions based on that specific information, reducing hallucinations.
    *   *What is the problem with standard RAG?* The LLM often ignores the retrieved documents and relies on its pre-training (hallucinating or giving generic answers). Also, the retrieval step and the generation step are usually trained separately.
    *   *What does ORPO do?* It combines the retrieval step and the generation step into a single, unified training process. It uses "preference optimization" (like RLHF - Reinforcement Learning from Human Feedback) to teach the model to *prefer* answers that are grounded in the retrieved documents.
    *   *Key mechanism:* It uses a single loss function that simultaneously optimizes the model to retrieve relevant documents *and* generate accurate answers based on them.

3.  **Translate to Simple Terms (The Analogy Strategy):**
    *   *Analogy idea:* A student taking an open-book exam.
    *   *Standard RAG:* The student has a textbook (retrieval) and a brain (generation), but they are taught separately. The student might just guess from memory instead of looking at the book.
    *   *ORPO:* The teacher trains the student *while* they are looking at the book, rewarding them specifically for using the book to answer correctly. The student learns to look up the right page *and* write the right answer at the same time.

4.  **Drafting the Explanation - Step-by-Step:**

    *   **The Hook/Definition:** Start with a one-sentence summary.
        *   *Draft:* ORPO is a way to train AI to better use outside information when answering questions.

    *   **The Context (What is RAG?):
"""

"""
(base) root@bbc73cbc4f50:~# python load8.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2026.4.8: Fast Qwen3_5 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce RTX 4090. Num GPUs = 1. Max memory: 47.371 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 8.9. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1184/1184 [00:19<00:00, 61.33it/s]
Skipping model.visual.merger.linear_fc1: no quant_state found
Skipping model.visual.merger.linear_fc2: no quant_state found
=== MODEL OUTPUT ===
1.  **Analyze the User's Input:**
    *   Input: "你好." (Hello.)
    *   Language: Chinese
    *   Intent: Greeting, initiating a conversation.

2.  **Determine the Appropriate Response:**
    *   Acknowledge the greeting warmly.
    *   Respond in the same language (Chinese).
    *   Offer assistance or ask how I can help.

3.  **Drafting the Response (Internal Monologue/Trial):**
    *   *Option 1:* 你好！有什么我可以帮你的吗？ (Hello! Is there anything I can help you with?) - *Standard, polite.*
    *   *Option 2:* 你好！很高兴见到你。今天有什么我可以为你做的吗？ (Hello! Nice to see you. Is there anything I can do for you today?) - *A bit more enthusiastic.*
    *   *Option 3:* 你好！请问有什么我可以帮您的？ (Hello! How can I help you?) - *Slightly more formal.*

4.  **Selecting the Best Response:**
    *   Option 1 is a great balance of friendly and helpful. Let's go with a variation of that.

5.  **Final Polish (Chinese):**
    *   "你好！很高兴和你聊天。请问有什么我可以帮你的吗？" (Hello! Nice to chat with you. Is there anything I can help you with?)

6.  **Output Generation:** (Translate the polished thought into the final output).
</think>

你好！很高兴和你聊天。请问有什么我可以帮你的吗？
"""
