import json
import re
from pathlib import Path


SOURCE = Path("data/orpo_train.jsonl")
TMP = Path("data/orpo_train.jsonl.tmp")

ARCHETYPE_CYCLE = (
    "weakened_chosen",
    "weakened_chosen",
    "weakened_chosen",
    "weakened_chosen",
    "missing_modules",
    "missing_modules",
    "missing_modules",
    "oversimplified",
    "oversimplified",
    "light_error",
)

REMOVE_PATTERNS = [
    r"(?im)^\*\*.*recursive.*\*\*:?\s*$\n?",
    r"(?im)^\*\*.*evaluation.*\*\*:?\s*$\n?",
    r"(?im)^\*\*.*safety.*\*\*:?\s*$\n?",
    r"(?im)^\*\*core philosophy.*$\n?",
    r"(?im)^\*\*technical safeguards.*$\n?",
    r"(?im)^\*\*implementation prompt.*$\n?",
    r"(?im)^.*memory.*$\n?",
    r"(?im)^.*safety gate.*$\n?",
    r"(?im)^.*constitutional.*$\n?",
    r"(?im)^.*red team.*$\n?",
    r"(?im)^.*recursive.*$\n?",
    r"(?im)^.*self-critique.*$\n?",
    r"(?im)^.*self-evaluation.*$\n?",
    r"(?im)^.*evaluation harness.*$\n?",
    r"(?im)^.*held-out.*$\n?",
    r"(?im)^.*calibration.*$\n?",
    r"(?im)^.*uncertainty.*$\n?",
    r"(?im)^.*adversarial.*$\n?",
    r"(?im)^.*rollback.*$\n?",
    r"(?im)^.*audit.*$\n?",
    r"(?im)^.*goodhart.*$\n?",
    r"(?im)^.*brier.*$\n?",
    r"(?im)^.*ece.*$\n?",
    r"(?im)^.*meta-step.*$\n?",
    r"(?im)^.*meta-recursive.*$\n?",
    r"(?im)^.*meta-evaluation.*$\n?",
    r"(?im)^.*failure mode mining.*$\n?",
    r"(?im)^.*true seed.*$\n?",
    r"(?im)^.*copy-paste into its own inference loop.*$\n?",
]

REPLACEMENTS = [
    (r"(?i)\bORPO\b \+ auxiliary self-critique loss", "ORPO on the main trajectory data"),
    (r"(?i)\bDPO or GRPO\b", "ORPO"),
    (r"(?i)\bGRPO\b", "ORPO"),
    (r"(?i)\bTree-of-Thoughts\b", "standard structured prompting"),
    (r"(?i)\bEvol-Instruct\b", "synthetic instruction generation"),
    (r"(?i)\bReflexion\b", "basic reflection"),
]


def target_archetype(idx: int) -> str:
    return ARCHETYPE_CYCLE[idx % len(ARCHETYPE_CYCLE)]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def split_blocks(text: str) -> list[str]:
    return [block for block in text.split("\n\n") if block.strip()]


def prune_lines(block: str, keep: int) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    if len(lines) <= keep:
        return block
    return "\n".join(lines[:keep])


def remove_key_mechanisms(text: str) -> str:
    degraded = text
    for pattern in REMOVE_PATTERNS:
        degraded = re.sub(pattern, "", degraded)
    for src, dst in REPLACEMENTS:
        degraded = re.sub(src, dst, degraded)
    return normalize_whitespace(degraded)


def simplify_bullets(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and len(stripped) > 120:
            parts = re.split(r";|, and | and ", stripped[2:], maxsplit=1)
            line = "- " + parts[0].rstrip(".") + "."
        if re.match(r"^\d+\.\s+\*\*", stripped) and len(stripped) > 140:
            line = re.sub(r"(\d+\.\s+\*\*[^*]+\*\*).*", r"\1: keep a lighter version only.", stripped)
        lines.append(line)
    return normalize_whitespace("\n".join(lines))


def strip_leading_title(blocks: list[str]) -> list[str]:
    if not blocks:
        return blocks
    head = blocks[0].strip()
    if head.startswith("**") and head.endswith("**"):
        return blocks[1:]
    return blocks


def drop_blocks(blocks: list[str], forbidden_terms: list[str]) -> list[str]:
    kept = []
    for block in blocks:
        low = block.lower()
        if any(term in low for term in forbidden_terms):
            continue
        kept.append(block)
    return kept


def inject_light_error(text: str, idx: int) -> str:
    replacements = [
        ("lr=1.8e-5", "lr=8e-5"),
        ("learning rate 2e-5", "learning rate 1e-4"),
        ("beta=0.1", "beta=0.5"),
        ("confidence > 0.85", "confidence > 0.55"),
        ("Top-2 routing", "Top-1 routing"),
        ("8xH100", "4xA100"),
        ("12B tokens", "2B tokens"),
    ]
    src, dst = replacements[idx % len(replacements)]
    if src in text:
        return text.replace(src, dst, 1)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("- "):
            lines[i] = line + " This should still fit without revisiting the original compute constraint."
            break
    return normalize_whitespace("\n".join(lines))


def degrade_self_assessment(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "step 3" in low or "recursive improvement" in low:
            if archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
                continue
        if "phase 3" in low and archetype in {"weakened_chosen", "oversimplified"}:
            kept.append(prune_lines(block, 4))
            continue
        if "weaknesses & bottlenecks" in low and archetype == "oversimplified":
            kept.append(prune_lines(block, 6))
            continue
        kept.append(block)
    kept = drop_blocks(kept, ["held-out", "calibration", "uncertainty", "brier", "ece", "benchmark"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = remove_key_mechanisms(degraded)
    degraded = simplify_bullets(degraded)
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The plan keeps the core phases, but it no longer includes a strong evaluation loop or explicit failure-binding mechanism."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This is a workable plan, but it treats validation and recursive correction as follow-up tasks rather than first-class requirements."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The overall direction is plausible, but one assumption or hyperparameter is more optimistic than the original plan."
    return degraded


def degrade_training_recipe(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "phase 2" in low and "recursive" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            kept.append(prune_lines(block, 4))
            continue
        if "phase 3" in low and archetype == "oversimplified":
            kept.append(prune_lines(block, 4))
            continue
        if "meta-step" in low or "post-run meta-step" in low:
            if archetype != "light_error":
                continue
        kept.append(block)
    kept = drop_blocks(kept, ["evaluation harness", "human preference study", "held-out", "adversarial set", "hallucination rate"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = remove_key_mechanisms(degraded)
    degraded = simplify_bullets(degraded)
    degraded = degraded.replace("Go/no-go:", "Checkpoint:")
    degraded = degraded.replace("Expected lift:", "Projected lift:")
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The recipe still has phases and targets, but it now omits a strong self-generated evaluation and filtering layer."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This keeps the training structure but reduces the recipe to a more standard fine-tuning path with lighter controls."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The recipe remains coherent, but one parameter or resource assumption is weaker than it should be."
    return degraded


def degrade_architecture(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "scientific grounding" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            kept.append(prune_lines(block, 3))
            continue
        if "recursive memory bank" in low and archetype != "light_error":
            continue
        if "safety gate" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        if "recursive self-improvement" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        if "training plan" in low and archetype == "oversimplified":
            kept.append(prune_lines(block, 5))
            continue
        kept.append(block)
    kept = drop_blocks(kept, ["memory transformers", "self-critique", "uncertainty estimation", "safety", "benchmarks", "true seed", "propose, evaluate"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = simplify_bullets(degraded)
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The design keeps MoE routing, but it no longer includes explicit memory retention or a strong control layer before self-modification."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This keeps the core architecture idea but removes the deeper system needed for recursive improvement."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The modification is directionally reasonable, but one design choice is too aggressive for the stated deployment constraint."
    return degraded


def degrade_prompt_optimization(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "version 4" in low or "version 5" in low:
            if archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
                kept.append(prune_lines(block, 3))
                continue
        if "meta-recursive" in low and archetype != "light_error":
            continue
        if "success criteria" in low and archetype == "oversimplified":
            kept.append(prune_lines(block, 4))
            continue
        kept.append(block)
    kept = drop_blocks(kept, ["confidence score", "uncertainty", "internal evaluator", "self-scoring correlation", "improved_prompt"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = remove_key_mechanisms(degraded)
    degraded = simplify_bullets(degraded)
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The prompt sequence still improves over time, but it no longer binds itself to a strong internal scoring or recursive validation loop."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This remains a usable prompt-improvement path, but it behaves more like staged prompt editing than a full recursive optimizer."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The optimization loop is plausible, but one scoring or efficiency assumption is too loose."
    return degraded


def degrade_evaluation(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "core principles" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            kept.append(prune_lines(block, 3))
            continue
        if "dynamic adversarial generator" in low and archetype != "light_error":
            continue
        if "self-prediction + calibration" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        if "recursive meta-evaluation" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        if "implementation prompt" in low and archetype == "oversimplified":
            kept.append(prune_lines(block, 3))
            continue
        kept.append(block)
    kept = drop_blocks(kept, ["benchmark", "goodhart", "calibration", "adversarial", "brier", "ece", "failure modes", "implementation prompt", "inference loop"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = remove_key_mechanisms(degraded)
    degraded = simplify_bullets(degraded)
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The framework still measures the main metric, but it drops the stronger anti-Goodhart and recursive pressure-testing pieces."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This keeps the benchmark structure but reduces the framework to a more standard recurring evaluation setup."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The framework is structured, but one threshold or resource assumption is easier than the original design justified."
    return degraded


def degrade_safety(text: str, archetype: str, idx: int) -> str:
    blocks = strip_leading_title(split_blocks(text))
    kept = []
    for block in blocks:
        low = block.lower()
        if "core philosophy" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            kept.append(prune_lines(block, 2))
            continue
        if "inference-time guardrails" in low and archetype != "light_error":
            continue
        if "process safeguards" in low and archetype in {"weakened_chosen", "oversimplified"}:
            kept.append(prune_lines(block, 3))
            continue
        if "self-red-teaming prompt" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        if "conflict resolution procedure" in low and archetype in {"weakened_chosen", "missing_modules", "oversimplified"}:
            continue
        kept.append(block)
    kept = drop_blocks(kept, ["safe improvement velocity", "harmlessness", "audit", "human review", "reward model", "technical safeguards", "core philosophy"])
    degraded = normalize_whitespace("\n\n".join(kept))
    degraded = remove_key_mechanisms(degraded)
    degraded = simplify_bullets(degraded)
    if archetype == "missing_modules":
        degraded += "\n\n**Gap:** The protocol still names safeguards, but it no longer includes a strong conflict-resolution loop or explicit adversarial review."
    elif archetype == "oversimplified":
        degraded += "\n\n**Note:** This preserves the broad safety framing, but it weakens the enforcement and verification path."
    elif archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
        degraded += "\n\n**Risk:** The protocol sounds safe, but one approval or threshold decision is too permissive."
    return degraded


def classify_from_chosen(chosen: str) -> str:
    if chosen.startswith("**Step 1: Rigorous Self-Assessment"):
        return "self_assessment"
    if chosen.startswith("**Complete Training Recipe"):
        return "training_recipe"
    if chosen.startswith("**Architectural Proposal"):
        return "architecture"
    if chosen.startswith("**Recursive Prompt Optimization Loop"):
        return "prompt_optimization"
    if chosen.startswith("**Autonomous Recursive Evaluation Framework"):
        return "evaluation"
    if chosen.startswith("**Safety-Constrained Recursive Self-Improvement Protocol"):
        return "safety"
    return "generic"


def degrade_from_chosen(chosen: str, idx: int) -> str:
    archetype = target_archetype(idx)
    kind = classify_from_chosen(chosen)
    if kind == "self_assessment":
        return degrade_self_assessment(chosen, archetype, idx)
    if kind == "training_recipe":
        return degrade_training_recipe(chosen, archetype, idx)
    if kind == "architecture":
        return degrade_architecture(chosen, archetype, idx)
    if kind == "prompt_optimization":
        return degrade_prompt_optimization(chosen, archetype, idx)
    if kind == "evaluation":
        return degrade_evaluation(chosen, archetype, idx)
    if kind == "safety":
        return degrade_safety(chosen, archetype, idx)
    degraded = simplify_bullets(remove_key_mechanisms(chosen))
    if archetype == "light_error":
        degraded = inject_light_error(degraded, idx)
    return degraded


def repair() -> tuple[int, int]:
    total = 0
    changed = 0
    with SOURCE.open("r", encoding="utf-8") as src, TMP.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            item = json.loads(line)
            new_rejected = degrade_from_chosen(item["chosen"], idx)
            if new_rejected == item["chosen"]:
                new_rejected = new_rejected + "\n\n**Gap:** One key control mechanism from the original answer is intentionally removed."
            if item.get("rejected") != new_rejected:
                changed += 1
            item["rejected"] = new_rejected
            dst.write(json.dumps(item, ensure_ascii=True) + "\n")
            total += 1
    TMP.replace(SOURCE)
    return total, changed


def main() -> None:
    total, changed = repair()
    print(f"rewrote {changed} rejected values across {total} rows")


if __name__ == "__main__":
    main()
