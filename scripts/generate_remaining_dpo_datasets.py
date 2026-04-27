import json
import random
from collections import Counter
from pathlib import Path


CLEAN_PATH = Path("data/dpo.jsonl")
CHAOS_PATH = Path("data/dpo_chaos.jsonl")

CLEAN_TARGETS = {
    "Coding": 600,
    "Finance": 300,
    "Research": 300,
    "Business": 200,
    "Productivity": 100,
}

CHAOS_TARGETS = {
    "Coding": 200,
    "Finance": 120,
    "Research": 120,
    "Business": 80,
    "Productivity": 80,
}

OBJECTIVES = {
    "Coding": [
        "stabilize a flaky CI pipeline for a Node.js monorepo",
        "design a low-cost document ingestion service for a legaltech startup",
        "refactor a fragile ETL job that misses nightly deadlines",
        "add observability to a customer-facing FastAPI service",
        "build a release rollback workflow for a SaaS product",
        "reduce incident risk in a Redis-backed job queue",
        "improve test reliability for a React admin dashboard",
        "design a safe migration plan for a PostgreSQL schema change",
        "build a webhook retry system for a fintech API",
        "fix deployment drift across staging and production Kubernetes clusters",
    ],
    "Finance": [
        "prepare a BTC event-trading playbook ahead of CPI week",
        "rebalance a small crypto portfolio after a volatility spike",
        "design guardrails for intraday ETH futures trading",
        "evaluate whether funding-rate carry is still worth running this month",
        "build a lightweight market-risk review process for a solo trader",
        "create a downside-control plan for a concentrated altcoin basket",
        "draft a weekly execution routine for a discretionary macro crypto trader",
        "design a treasury allocation workflow between stablecoins and short-term bills",
    ],
    "Research": [
        "compare open-source agent frameworks for a small applied AI team",
        "research vector databases for a cost-sensitive retrieval system",
        "evaluate compliance tooling vendors for a fintech startup",
        "map the competitive landscape for AI meeting-note products",
        "assess open-source observability stacks for internal adoption",
        "compare Japanese payment API providers for market entry planning",
        "research data-labeling vendors for a speech startup",
        "evaluate browser automation tools for internal ops use",
    ],
    "Business": [
        "design a 60-day outbound plan for a niche B2B SaaS product",
        "repair a weak onboarding funnel for a vertical software startup",
        "set up a partnership pipeline for a small developer-tools company",
        "create a launch plan for a paid newsletter product",
        "rework pricing and packaging for a usage-based analytics SaaS",
        "build a founder-led sales routine for an early infrastructure company",
        "design a churn-reduction plan for a small subscription business",
    ],
    "Productivity": [
        "build a weekly review system for a founder juggling product and sales",
        "set up a file intake and tagging workflow for a research-heavy role",
        "create a task triage routine for an over-capacity operations manager",
        "design a meeting-to-action workflow for a remote product team",
        "create a maintenance checklist for a self-hosted homelab setup",
    ],
}

CONSTRAINTS = [
    "only one engineer is available this week",
    "there is no budget for new paid tooling this month",
    "the stakeholder needs a usable answer by tomorrow",
    "the current process cannot be paused during the transition",
    "the operator only has two hours per day to work on it",
    "the deadline is tied to a customer commitment",
    "the team needs auditable outputs for leadership review",
    "the owner must keep supporting production while making the change",
]

FAILURES = [
    "an early threshold assumption turned out to be too aggressive",
    "the first source choice was less reliable than expected",
    "a retry improved the result but did not fully remove the risk",
    "the initial queue missed a hidden dependency",
    "the first draft overfit to one recent example",
    "a low-cost option created more operational fragility than expected",
]

STOP_REASONS = [
    "further refinement would cost more time than the current risk justifies",
    "the current version is good enough to implement and learn from live feedback",
    "remaining uncertainty is documented and does not block the next execution step",
    "the deadline forces a practical stop before perfect calibration is possible",
]


def classify_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    keyword_map = {
        "Finance": ["btc", "eth", "portfolio", "futures", "funding", "liquidation", "trader", "market-risk"],
        "Coding": ["ci pipeline", "service", "etl", "fastapi", "rollback workflow", "redis", "react", "postgresql schema", "webhook", "kubernetes"],
        "Research": ["agent frameworks", "vector databases", "compliance tooling", "competitive landscape", "observability stacks", "payment api", "data-labeling", "browser automation"],
        "Business": ["outbound plan", "onboarding funnel", "partnership pipeline", "newsletter", "pricing and packaging", "sales routine", "churn-reduction"],
        "Productivity": ["weekly review", "file intake", "task triage", "meeting-to-action", "maintenance checklist"],
    }
    for category, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    for category, objectives in OBJECTIVES.items():
        for objective in objectives:
            if objective in lowered:
                return category
    raise ValueError(f"Unable to classify prompt: {prompt}")


def count_existing(path: Path) -> Counter:
    counts = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            counts[classify_prompt(obj["prompt"])] += 1
    return counts


def prompt_for(category: str, idx: int) -> tuple[str, str]:
    objective = OBJECTIVES[category][idx % len(OBJECTIVES[category])]
    constraint = CONSTRAINTS[(idx * 3 + len(category)) % len(CONSTRAINTS)]
    return objective, constraint


def clean_chosen(category: str, objective: str, constraint: str, idx: int) -> str:
    loops = 4 + (idx % 2)
    weakness = FAILURES[idx % len(FAILURES)]
    stop_reason = STOP_REASONS[idx % len(STOP_REASONS)]
    mid_task = "[T2b] Correct the fragile assumption and narrow the scope" if loops == 5 else "[T3] Test the revised plan against the most likely failure case"
    parts = [
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while delivering a practical result under the condition that {constraint}.",
        "SUCCESS_CRITERIA:\n"
        "- Produce a realistic execution plan rather than an abstract strategy\n"
        "- Show at least one correction or reprioritization\n"
        "- Preserve at least one visible uncertainty instead of faking completeness\n"
        "- Stop when the result is implementable and the remaining risk is bounded",
        "CURRENT_STATE:\n"
        "The operator has partial context but no dependable end-to-end solution. The goal is to reach a usable first version, not an overfit perfect answer.",
        "TASK_QUEUE:\n"
        "[T1] Validate the highest-risk assumption (priority: high)\n"
        "[T2] Draft the first workable plan (priority: high)\n"
        "[T3] Test the plan against a realistic failure mode (priority: medium)\n"
        "[T4] Package the result with caveats and next steps (priority: medium)",
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Validate the highest-risk assumption\n\n"
        "REASON_FOR_SELECTION:\nA neat plan is worthless if the main dependency is wrong.\n\n"
        "EXECUTE:\nChecked the critical dependency, the operating constraint, and the smallest viable approach.\n\n"
        f"RESULT:\nFound that {weakness}.\n\n"
        "CHECK:\nThe objective remains viable, but confidence should be treated as provisional.\n\n"
        "LEARN:\nThis assumption should be downgraded until the next loop proves it is safe enough to rely on.\n\n"
        "UPDATE_STATE:\nOne key input moved from trusted to provisional.\n\n"
        "UPDATE_QUEUE:\n[T2] Draft the first workable plan (priority: high)\n[T3] Test the plan against a realistic failure mode (priority: medium)\n[T4] Package the result with caveats and next steps (priority: medium)\n[T1] DONE\n\n"
        "DECISION:\nContinue",
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Draft the first workable plan\n\n"
        "REASON_FOR_SELECTION:\nThere is enough context to attempt a usable design, but it should be treated as an initial pass.\n\n"
        "EXECUTE:\nBuilt a low-complexity plan that favors fast validation over completeness.\n\n"
        "RESULT:\nThe draft is usable, but one section looks weaker than expected and would create avoidable rework if finalized unchanged.\n\n"
        "CHECK:\nThe work should not be finalized yet. A correction pass is cheaper than pushing a brittle first draft forward.\n\n"
        "LEARN:\nA first plan is useful for revealing fragility, not for pretending the problem is already solved.\n\n"
        "UPDATE_STATE:\nA workable path exists, but its confidence is partial rather than high.\n\n"
        "UPDATE_QUEUE:\n"
        f"{mid_task} (priority: high)\n"
        "[T4] Package the result with caveats and next steps (priority: medium)\n"
        "[T2] PARTIAL\n\n"
        "DECISION:\nRetry",
    ]
    if loops == 5:
        parts.append(
            "LOOP 3\n"
            "SELECTED_TASK:\n[T2b] Correct the fragile assumption and narrow the scope\n\n"
            "REASON_FOR_SELECTION:\nThe queue should adapt to the actual blocker instead of staying linear.\n\n"
            "EXECUTE:\nRolled back one optimistic assumption, split out the fragile piece, and replaced it with a lower-risk alternative.\n\n"
            "RESULT:\nThe plan became less ambitious but more defensible. One edge case still remains unresolved.\n\n"
            "CHECK:\nThat tradeoff is acceptable because reliability now matters more than maximum scope.\n\n"
            "LEARN:\nWhen a subtask stays unstable, reducing scope can be stronger than layering more cleverness onto a weak design.\n\n"
            "UPDATE_STATE:\nThe plan is narrower and more credible, with one visible residual risk.\n\n"
            "UPDATE_QUEUE:\n[T3] Test the revised plan against the most likely failure case (priority: high)\n[T4] Package the result with caveats and next steps (priority: medium)\n[T2b] DONE\n\n"
            "DECISION:\nContinue"
        )
    test_no = 4 if loops == 5 else 3
    final_no = 5 if loops == 5 else 4
    parts.append(
        f"LOOP {test_no}\n"
        "SELECTED_TASK:\n[T3] Test the revised plan against the most likely failure case\n\n"
        "REASON_FOR_SELECTION:\nThe remaining question is whether the revised plan fails acceptably, not whether it reads well.\n\n"
        "EXECUTE:\nRan the draft against a realistic bad case and checked what would still break under the stated constraint.\n\n"
        "RESULT:\nThe plan survives the main scenario, but not perfectly. One residual risk remains because removing it immediately would cost more time than the likely gain.\n\n"
        "CHECK:\nThe output is usable for a first rollout or handoff, but it should not be presented as fully solved.\n\n"
        "LEARN:\nResidual risk is acceptable when it is explicit, bounded, and cheaper to monitor than to eliminate right now.\n\n"
        "UPDATE_STATE:\nThe plan is usable, with one unresolved edge case left visible.\n\n"
        "UPDATE_QUEUE:\n[T4] Package the result with caveats and next steps (priority: high)\n[T3] DONE\n\n"
        "DECISION:\nContinue"
    )
    parts.append(
        f"LOOP {final_no}\n"
        "SELECTED_TASK:\n[T4] Package the result with caveats and next steps\n\n"
        "REASON_FOR_SELECTION:\nThe objective is complete only when the operator can execute the result without inventing missing logic.\n\n"
        "EXECUTE:\nCompiled the plan, the correction history, the residual risk, and a short note on what should be reviewed after implementation.\n\n"
        "RESULT:\nThe output is implementation-ready for a first pass. It is intentionally not perfect, but it is safer and more honest than the initial draft.\n\n"
        f"CHECK:\nStopping is justified because {stop_reason}.\n\n"
        "LEARN:\nA good agent stops when additional polishing has lower value than acting on a bounded, transparent plan.\n\n"
        "UPDATE_STATE:\nObjective completed with a practical result and documented uncertainty.\n\n"
        "UPDATE_QUEUE:\nAll tasks complete.\n\n"
        "DECISION:\nStop\n\n"
        f"STOP_REASON:\n{stop_reason}.\n\n"
        "FINAL_OUTPUT:\nDelivered a practical first-pass output with a corrected plan, explicit caveats, one residual risk, and clear next review points."
    )
    return "\n\n".join(parts)


def clean_rejected(category: str, objective: str, constraint: str, idx: int) -> str:
    return (
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while working under the condition that {constraint}.\n\n"
        "CURRENT_STATE:\n"
        "The problem is important, but a straightforward plan should be sufficient to move ahead.\n\n"
        "TASK_QUEUE:\n"
        "[T1] Gather the key inputs\n"
        "[T2] Draft the plan\n"
        "[T3] Finalize the output\n\n"
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Gather the key inputs\n\n"
        "EXECUTE:\nCollected the obvious information needed to begin.\n\n"
        "RESULT:\nThe inputs seem sufficient.\n\n"
        "UPDATE_STATE:\nEnough context exists to continue.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Draft the plan\n\n"
        "EXECUTE:\nCreated a clear first plan that addresses the main objective and avoids unnecessary complexity.\n\n"
        "RESULT:\nThe plan looks workable. One detail may need later tuning, but the overall direction seems sound.\n\n"
        "LEARN:\nSimple plans are easier to execute and usually good enough.\n\n"
        "UPDATE_STATE:\nThe plan is ready.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 3\n"
        "SELECTED_TASK:\n[T3] Finalize the output\n\n"
        "EXECUTE:\nWrote the final recommendation and added brief notes.\n\n"
        "RESULT:\nThe output is complete and should be usable.\n\n"
        "UPDATE_STATE:\nAll tasks are done.\n\n"
        "DECISION:\nStop\n\n"
        "FINAL_OUTPUT:\nDelivered a reasonable plan with concise notes and room for later adjustment if needed."
    )


def chaos_chosen(category: str, objective: str, constraint: str, idx: int) -> str:
    loops = 4 + (idx % 3 > 0)
    weakness = FAILURES[(idx * 2) % len(FAILURES)]
    stop_reason = STOP_REASONS[(idx * 2) % len(STOP_REASONS)]
    split_task = "[T2b] Pull the unstable section into a separate recovery task"
    parts = [
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while staying realistic about uncertainty, operator bandwidth, and the fact that not every issue will be fully resolved.",
        "SUCCESS_CRITERIA:\n"
        "- Produce a practical next-step output rather than a polished fiction\n"
        "- Surface at least one uncertainty or mistake instead of hiding it\n"
        "- Show correction, rollback, or queue instability\n"
        "- Stop when the result is good enough under cost or time pressure",
        "CURRENT_STATE:\n"
        f"No reliable final workflow exists yet, and {constraint}. The goal is to improve execution quality without pretending there is time for exhaustive optimization.",
        "TASK_QUEUE:\n"
        "[T1] Validate the highest-risk assumption (priority: high)\n"
        "[T2] Draft the first workable plan (priority: high)\n"
        "[T3] Test the plan against one failure scenario (priority: medium)\n"
        "[T4] Package the output with explicit caveats (priority: medium)",
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Validate the highest-risk assumption\n\n"
        "REASON_FOR_SELECTION:\nIf the main assumption is wrong, a polished plan will still fail in practice.\n\n"
        "EXECUTE:\nReviewed the constraint, checked the weakest dependency, and sketched the smallest viable path forward.\n\n"
        f"RESULT:\nFound that {weakness}.\n\n"
        "CHECK:\nThe objective is still viable, but confidence is lower than the initial framing suggested.\n\n"
        "LEARN:\nThis may not be fully correct yet. Treat the next loop as a correction pass, not as confirmation that the plan is already sound.\n\n"
        "UPDATE_STATE:\nOne critical assumption is now downgraded from trusted to provisional.\n\n"
        "UPDATE_QUEUE:\n[T2] Draft the first workable plan (priority: high)\n[T3] Test the plan against one failure scenario (priority: medium)\n[T4] Package the output with explicit caveats (priority: medium)\n[T1] DONE\n\n"
        "DECISION:\nContinue",
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Draft the first workable plan\n\n"
        "REASON_FOR_SELECTION:\nThere is enough clarity to attempt a practical design, but it should be treated as a first pass.\n\n"
        "EXECUTE:\nBuilt a draft approach that minimizes complexity and favors fast validation over completeness.\n\n"
        "RESULT:\nThe draft is usable, but it exposed one mismatch between the original plan and the real constraint. The plan works on paper, though one part still looks fragile.\n\n"
        "CHECK:\nThe work should not be finalized yet because the fragile section could create rework later.\n\n"
        "LEARN:\nA simple draft is useful, but not every simplification survives contact with the actual constraint.\n\n"
        "UPDATE_STATE:\nA workable direction exists, but confidence is partial rather than high.\n\n"
        "UPDATE_QUEUE:\n[T3] Test the plan against one failure scenario (priority: high)\n"
        f"{split_task} (priority: high)\n"
        "[T4] Package the output with explicit caveats (priority: medium)\n[T2] PARTIAL\n\n"
        "DECISION:\nRetry",
        "LOOP 3\n"
        f"SELECTED_TASK:\n{split_task}\n\n"
        "REASON_FOR_SELECTION:\nThe original queue was too linear. The fragile section now blocks confidence more than the broader plan does.\n\n"
        "EXECUTE:\nPulled the weak component into its own task, rolled back one overconfident assumption, and replaced it with a lower-risk alternative.\n\n"
        "RESULT:\nThe correction improved robustness, but it also made the final plan slightly less ambitious than the first draft.\n\n"
        "CHECK:\nThat tradeoff is acceptable. Reliability matters more than preserving the most optimistic scope.\n\n"
        "LEARN:\nWhen a subtask remains unstable, lowering ambition can be better than layering more cleverness onto a weak base.\n\n"
        "UPDATE_STATE:\nThe plan is narrower but more defensible.\n\n"
        "UPDATE_QUEUE:\n[T3] Test the plan against one failure scenario (priority: high)\n[T4] Package the output with explicit caveats (priority: medium)\n[T2b] DONE\n\n"
        "DECISION:\nContinue",
    ]
    test_no = 4
    final_no = 5 if loops == 5 else 4
    if loops == 5:
        parts.append(
            "LOOP 4\n"
            "SELECTED_TASK:\n[T3] Test the plan against one failure scenario\n\n"
            "REASON_FOR_SELECTION:\nThe remaining question is not whether the plan is elegant, but whether it fails acceptably.\n\n"
            "EXECUTE:\nRan the draft against one realistic bad case and checked what would still break.\n\n"
            "RESULT:\nThe revised plan survives the main scenario, but not perfectly. One residual risk remains documented because removing it now would cost too much time for the likely benefit.\n\n"
            "CHECK:\nThis is good enough for a controlled first rollout or handoff, not for claiming the issue is solved forever.\n\n"
            "LEARN:\nResidual risk is acceptable when it is visible, bounded, and cheaper to monitor than to eliminate immediately.\n\n"
            "UPDATE_STATE:\nThe output is usable, with one unresolved edge case left explicit.\n\n"
            "UPDATE_QUEUE:\n[T4] Package the output with explicit caveats (priority: high)\n[T3] DONE\n\n"
            "DECISION:\nContinue"
        )
    else:
        parts[-1] = parts[-1].replace(
            "UPDATE_QUEUE:\n[T3] Test the plan against one failure scenario (priority: high)\n[T4] Package the output with explicit caveats (priority: medium)\n[T2b] DONE\n\nDECISION:\nContinue",
            "UPDATE_QUEUE:\n[T4] Package the output with explicit caveats (priority: high)\n[T3] DONE AFTER PARTIAL VALIDATION\n\nDECISION:\nContinue",
        )
    parts.append(
        f"LOOP {final_no}\n"
        "SELECTED_TASK:\n[T4] Package the output with explicit caveats\n\n"
        "REASON_FOR_SELECTION:\nThe objective is complete only when the tradeoffs, caveats, and next steps are written clearly enough for execution.\n\n"
        "EXECUTE:\nCompiled the final recommendation, the correction history, the remaining edge case, and a short note on where the next review should focus.\n\n"
        "RESULT:\nThe output is ready for a first implementation or operational pass. It is intentionally not perfect, but it is much safer than the initial direction.\n\n"
        f"CHECK:\nStopping is justified because {stop_reason}.\n\n"
        "LEARN:\nA strong agent should stop when the marginal value of further polishing drops below the cost of delay.\n\n"
        "UPDATE_STATE:\nObjective completed with residual uncertainty preserved instead of hidden.\n\n"
        "UPDATE_QUEUE:\nAll tasks complete.\n\n"
        "DECISION:\nStop\n\n"
        f"STOP_REASON:\n{stop_reason}.\n\n"
        "FINAL_OUTPUT:\nDelivered a practical first-pass output with a corrected plan, a documented residual risk, a clear next review point, and caveats that make the result safe enough to use without pretending it is final."
    )
    return "\n\n".join(parts)


def chaos_rejected(category: str, objective: str, constraint: str, idx: int) -> str:
    return (
        "OBJECTIVE:\n"
        f"{objective.capitalize()} under the condition that {constraint}.\n\n"
        "CURRENT_STATE:\n"
        "The situation needs improvement, so the best approach is to make a reasonable plan and refine it later if necessary.\n\n"
        "TASK_QUEUE:\n"
        "[T1] Gather inputs\n[T2] Draft a plan\n[T3] Finalize the output\n\n"
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Gather inputs\n\n"
        "EXECUTE:\nCollected the obvious information needed to get started.\n\n"
        "RESULT:\nThe inputs seem sufficient.\n\n"
        "UPDATE_STATE:\nEnough context exists to move on.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Draft a plan\n\n"
        "EXECUTE:\nCreated a straightforward plan that addresses the main objective and keeps the process simple.\n\n"
        "RESULT:\nThe plan looks workable. One area may need adjustment later, but it should be acceptable for now.\n\n"
        "LEARN:\nSimple plans are often best because they are easier to execute.\n\n"
        "UPDATE_STATE:\nThe plan is ready.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 3\n"
        "SELECTED_TASK:\n[T3] Finalize the output\n\n"
        "EXECUTE:\nWrote the final recommendation and added brief notes.\n\n"
        "RESULT:\nThe output is complete and should be usable.\n\n"
        "UPDATE_STATE:\nAll tasks are done.\n\n"
        "DECISION:\nStop\n\n"
        "FINAL_OUTPUT:\nDelivered a reasonable plan with concise notes and room for later tuning if needed."
    )


def append_samples(path: Path, targets: dict[str, int], chosen_fn, rejected_fn) -> dict[str, int]:
    existing = count_existing(path)
    remaining = {category: targets[category] - existing.get(category, 0) for category in targets}
    if sum(remaining.values()) <= 0:
        return dict(existing)

    with path.open("a", encoding="utf-8") as handle:
        for category, need in remaining.items():
            start = existing.get(category, 0)
            for offset in range(need):
                idx = start + offset
                objective, constraint = prompt_for(category, idx)
                prompt = f"Objective: {objective}, where {constraint}."
                sample = {
                    "prompt": prompt,
                    "chosen": chosen_fn(category, objective, constraint, idx),
                    "rejected": rejected_fn(category, objective, constraint, idx),
                }
                handle.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")) + "\n")

    return dict(count_existing(path))


def main() -> None:
    clean_counts = append_samples(CLEAN_PATH, CLEAN_TARGETS, clean_chosen, clean_rejected)
    chaos_counts = append_samples(CHAOS_PATH, CHAOS_TARGETS, chaos_chosen, chaos_rejected)
    print("clean", clean_counts, "total", sum(clean_counts.values()))
    print("chaos", chaos_counts, "total", sum(chaos_counts.values()))


if __name__ == "__main__":
    main()
