#!/usr/bin/env python3
"""
清洗 orpo_train.jsonl 的 rejected 字段，针对三个问题：
1. light_error 类型禁止保留整段 recursive/memory blocks
2. Gap/Note/Risk 文案换词，避免强信号词
3. 过短残片做最小长度重写
"""

import json
import re
import random

INPUT_FILE = '/home/yuki/Code/Llm/Versper-V1-Evo/data/orpo_train.jsonl'
OUTPUT_FILE = '/home/yuki/Code/Llm/Versper-V1-Evo/data/orpo_train_cleaned.jsonl'

# 强信号词列表，需要替换或移除
SIGNAL_WORDS = {
    'recursive': ['recursive', 'self-recursive', 'recursively', 'self-improving', 'self-improvement'],
    'evaluation': ['evaluation', 'eval', 'meta-eval', 'self-eval'],
    'safety': ['safety', 'guardrail', 'red team', 'constitutional'],
    'memory': ['memory bank', 'memory module', 'long-term memory', 'context memory'],
}

# 替换词映射
REPLACEMENTS = {
    'recursive': ['iterative', 'loop-based', 'stepwise'],
    'evaluation': ['assessment', 'check', 'verification'],
    'safety': ['policy', 'guidelines', 'oversight'],
    'memory': ['cache', 'store', 'buffer'],
}

# 章节标题中的强信号词
SECTION_SIGNALS = [
    r'Recursive\s+(Memory|Self-Improvement|Evaluation)',
    r'Self[- ]generated',
    r'Self[- ]improving',
    r'Meta[- ]evaluation',
    r'Red\s+team',
]


def clean_rejected(rejected: str, idx: int, prompt: str = "") -> str:
    """清洗单个 rejected"""
    result = rejected

    # 判断类型 (按原逻辑循环分配)
    bucket = idx % 4
    if bucket == 0:
        reject_type = 'weakened_chosen'
    elif bucket == 1:
        reject_type = 'missing_modules'
    elif bucket == 2:
        reject_type = 'oversimplified'
    else:
        reject_type = 'light_error'

    # 1. light_error 类型：删除整段 recursive/memory blocks
    if reject_type == 'light_error':
        # 删除 Recursive Memory Bank 整段
        result = re.sub(
            r'##\s+Recursive Memory Bank.*?(?=\n##|\n#|\Z)',
            '## Memory Storage\nA simple buffer stores recent interaction history.',
            result,
            flags=re.DOTALL
        )
        # 删除 Recursive Self-Improvement 整段
        result = re.sub(
            r'##\s+Recursive Self[- ]Improvement.*?(?=\n##|\n#|\Z)',
            '## Improvement Process\nApply standard optimization techniques.',
            result,
            flags=re.DOTALL
        )
        # 删除 Recursive Evaluation 整段
        result = re.sub(
            r'##\s+Recursive Meta[- ]Evaluation.*?(?=\n##|\n#|\Z)',
            '## Assessment\nUse standard benchmarks.',
            result,
            flags=re.DOTALL
        )

    # 2. Gap/Note/Risk 段落：替换强信号词
    for signal_key, words in SIGNAL_WORDS.items():
        replacements = REPLACEMENTS.get(signal_key, ['standard'])
        for word in words:
            if word.lower() in result.lower():
                # 替换但不是完全相同的词
                result = re.sub(
                    rf'\b{re.escape(word)}\b',
                    lambda m: random.choice(replacements),
                    result,
                    flags=re.IGNORECASE
                )

    # 3. 清理章节标题中的残留信号
    for pattern in SECTION_SIGNALS:
        result = re.sub(pattern, lambda m: 'Standard Process', result, flags=re.IGNORECASE)

    # 4. 处理过短残片
    lines = result.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过只标题没有内容的行
        if stripped and not stripped.startswith('#'):
            filtered_lines.append(line)
        elif stripped.startswith('#') and len(stripped) > 70:
            # 标题太长可能是残留
            continue
        elif stripped.startswith('#'):
            filtered_lines.append(line)

    result = '\n'.join(filtered_lines)

    # 最小长度检查 (< 200 字符的需要补全)
    if len(result) < 200:
        result = expand_short_rejected(result)

    # 清理多余空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()

    return result


def expand_short_rejected(short_rejected: str) -> str:
    """对过短的 rejected 进行扩展"""
    base = short_rejected.strip()

    # 如果已经有内容但太短，补全
    if len(base) > 50:
        additions = [
            "\n\nImplementation details:\n- Standard approach applies here\n- No special modules required\n- Basic monitoring suffices",
            "\n\nExpected outcome:\n- Incremental improvement expected\n- Standard benchmarks will show gains\n- No novel architecture needed",
            "\n\nRisks and mitigations:\n- Standard risks apply\n- Standard mitigation strategies\n- Regular assessment sufficient",
        ]
        base += random.choice(additions)

    return base


def main():
    processed = 0
    stats = {'recursive_hits': 0, 'eval_hits': 0, 'short': 0}

    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for idx, line in enumerate(fin):
            d = json.loads(line)
            prompt = d.get('prompt', '')
            rejected = d.get('rejected', '')

            cleaned = clean_rejected(rejected, idx, prompt)

            # 统计清洗后残留
            for kw in SIGNAL_WORDS['recursive']:
                if kw.lower() in cleaned.lower():
                    stats['recursive_hits'] += 1
                    break
            for kw in SIGNAL_WORDS['evaluation']:
                if kw.lower() in cleaned.lower():
                    stats['eval_hits'] += 1
                    break
            if len(cleaned) < 200:
                stats['short'] += 1

            d['rejected'] = cleaned
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')
            processed += 1

    print(f"Processed: {processed}")
    print(f"残留统计:")
    print(f"  recursive 命中: {stats['recursive_hits']}")
    print(f"  evaluation 命中: {stats['eval_hits']}")
    print(f"  过短 (<200): {stats['short']}")
    print(f"输出: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()