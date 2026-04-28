from unsloth import FastLanguageModel
import torch
import os

# 1. 加载模型和分词器/处理器
# 确保使用 load_in_4bit=True 保持量化状态
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/yijia-tmp/model",  # /root/checkpoint_4bit
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)

# 2. 修复 Tokenizer 配置
# 这样保存后的 checkpoint 也会包含这个修复设置
tokenizer.fix_mistral_regex = True

# 3. 设置保存路径
save_directory = "/root/checkpoint_4bit"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 4. 执行保存操作
# save_method="merged_4bit" 会保存为 Hugging Face 标准的 4-bit 权重文件夹
print(f"正在保存 4-bit 模型至: {save_directory}...")

model.save_pretrained_merged(
    save_directory,
    tokenizer,
    save_method="merged_4bit_forced",
)

print(f"保存成功！文件夹结构已生成在 {save_directory}")

"""
(base) root@bbc73cbc4f50:~# ls
'=1.4.0'   checkpoint_4bit   load.py   load2.py   load3.py   load4.py   load5.py   load6.py   load7.py   load8.py   merge_4bit.py   miniconda3   model_rest   unsloth_compiled_cache   yijia-tmp
(base) root@bbc73cbc4f50:~# ls checkpoint_4bit/
chat_template.jinja  generation_config.json            model-00002-of-00005.safetensors  model-00004-of-00005.safetensors  model.safetensors.index.json  tokenizer.json
config.json          model-00001-of-00005.safetensors  model-00003-of-00005.safetensors  model-00005-of-00005.safetensors  processor_config.json         tokenizer_config.json
"""
