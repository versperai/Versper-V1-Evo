#!/usr/bin/env python3
"""
批量清洗 rejected 中的问题
"""

import json
import re
import random

INPUT = '/home/yuki/Code/Llm/Versper-V1-Evo/data/orpo_train_fixed.jsonl'
OUTPUT = '/home/yuki/Code/Llm/Versper-V1-Evo/data/orpo_train_cleaned.jsonl'

# 需要替换的短语
REPLACE_RULES = [
    # Gap/pressure 替换
    (r'Gap:.*?(?=\n\n|\Z)', 'Gap: Simple measurement approach without advanced stress testing.'),
    (r'pressure[- ]test\w*', 'standard validation'),
    (r'recursive', 'iterative'),
    (r'self-improvement', 'standard optimization'),
    (r'self-generated', 'algorithmically generated'),
    (r'meta[- ]eval', 'assessment'),
    (r'red\s+team', 'review panel'),
]

# 残片扩展模板
EXPANSION_TEMPLATES = [
    "The framework uses basic measurement without advanced adversarial generation or self-assessment capabilities.",
    "Standard benchmark testing applies here. No recursive or adaptive components included.",
    "Uses simple periodic evaluation without custom test generation or meta-evaluation loops.",
]

def clean_rejected(text: str, idx: int) -> str:
    result = text.strip()
    
    # 1. 处理残片
    if result == '**Framework Components:**' or len(result) < 30:
        result = random.choice(EXPANSION_TEMPLATES)
        return result
    
    # 2. 替换短语
    for pattern, replacement in REPLACE_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # 3. 处理 Gap 行
    if 'Gap:' in result:
        # 只保留一行替换
        result = re.sub(r'Gap:.*', 'Gap: Simple metric tracking without advanced components.', result)
    
    # 4. 清理过多空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    # 5. 确保最小长度
    if len(result) < 150:
        addition = random.choice(EXPANSION_TEMPLATES)
        result = result + '\n\n' + addition
    
    return result.strip()

# 处理
fixed_count = 0
stats = {'too_short': 0, 'residue': 0}

with open(INPUT) as fin, open(OUTPUT, 'w') as fout:
    for idx, line in enumerate(fin):
        d = json.loads(line)
        old_rejected = d['rejected']
        
        cleaned = clean_rejected(old_rejected, idx)
        d['rejected'] = cleaned
        
        # 统计清洗后状态
        if len(cleaned) < 200:
            stats['too_short'] += 1
        
        # 检查残留
        if 'recursive' in cleaned.lower() or 'gap:' in cleaned.lower():
            stats['residue'] += 1
        
        fout.write(json.dumps(d, ensure_ascii=False) + '\n')
        fixed_count += 1

print(f'Processed: {fixed_count}')
print(f'Stats after clean: too_short={stats["too_short"]}, residue={stats["residue"]}')
print(f'Output: {OUTPUT}')