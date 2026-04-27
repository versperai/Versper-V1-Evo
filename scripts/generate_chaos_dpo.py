import json
import random
from pathlib import Path


OUTPUT = Path("data/dpo_chaos.jsonl")
TOTAL = 25

CATEGORIES = [
    ("Coding", 8),
    ("Finance", 5),
    ("Research", 5),
    ("Business", 4),
    ("Productivity", 3),
]

DOMAINS = {
    "Coding": [
        "stabilize a flaky CI pipeline for a Node.js monorepo",
        "design a low-cost document ingestion service for a legaltech startup",
        "refactor a fragile ETL job that misses nightly deadlines",
        "add observability to a customer-facing FastAPI service",
        "build a release rollback workflow for a SaaS product",
        "reduce incident risk in a Redis-backed job queue",
        "improve test reliability for a React admin dashboard",
        "design a safe migration plan for a PostgreSQL schema change",
    ],
    "Finance": [
        "prepare a BTC event-trading playbook ahead of CPI week",
        "rebalance a small crypto portfolio after a volatility spike",
        "design guardrails for intraday ETH futures trading",
        "evaluate whether funding-rate carry is still worth running this month",
        "build a lightweight market-risk review process for a solo trader",
    ],
    "Research": [
        "compare open-source agent frameworks for a small applied AI team",
        "research vector databases for a cost-sensitive retrieval system",
        "evaluate compliance tooling vendors for a fintech startup",
        "map the competitive landscape for AI meeting-note products",
        "assess open-source observability stacks for internal adoption",
    ],
    "Business": [
        "design a 60-day outbound plan for a niche B2B SaaS product",
        "repair a weak onboarding funnel for a vertical software startup",
        "set up a partnership pipeline for a small developer-tools company",
        "create a launch plan for a paid newsletter product",
    ],
    "Productivity": [
        "build a weekly review system for a founder juggling product and sales",
        "set up a file intake and tagging workflow for a research-heavy role",
        "create a task triage routine for an over-capacity operations manager",
    ],
}

CONSTRAINTS = [
    "only one engineer is available this week",
    "there is no budget for new paid tooling this month",
    "the stakeholder needs a usable answer by tomorrow",
    "the current process cannot be paused during the transition",
    "the operator only has two hours per day to work on it",
]

FAILURES = [
    "an early threshold assumption turned out to be too aggressive",
    "the first source choice was less reliable than expected",
    "a retry improved the result but did not fully remove the risk",
    "the initial queue missed a hidden dependency",
    "the first draft overfit to one recent example",
]

STOP_REASONS = [
    "further refinement would cost more time than the current risk justifies",
    "the current version is good enough to implement and learn from live feedback",
    "remaining uncertainty is documented and does not block the next execution step",
    "the deadline forces a practical stop before perfect calibration is possible",
]


def build_prompt(category: str, objective: str, constraint: str) -> str:
    return f"Objective: {objective}, where {constraint}."


def chosen_text(category: str, objective: str, constraint: str, idx: int) -> str:
    rnd = random.Random(f"chosen-{category}-{idx}")
    failure = FAILURES[idx % len(FAILURES)]
    stop_reason = STOP_REASONS[idx % len(STOP_REASONS)]
    loops = 4 if idx % 2 == 0 else 5
    queue_extra = "[T2b]" if loops == 5 else "[T4b]"
    sections = [
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while staying realistic about uncertainty, operator bandwidth, and the fact that not every issue will be fully resolved.",
        "SUCCESS_CRITERIA:\n"
        "- Produce a practical next-step plan rather than a theoretical ideal\n"
        "- Surface at least one key risk or uncertainty instead of hiding it\n"
        "- Show at least one correction, rollback, or reprioritization\n"
        "- Stop when the output is good enough under the stated constraint",
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
        f"RESULT:\nFound that {failure}.\n\n"
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
        f"UPDATE_QUEUE:\n[T3] Test the plan against one failure scenario (priority: high)\n{queue_extra} Split out the fragile part for targeted correction (priority: high)\n[T4] Package the output with explicit caveats (priority: medium)\n[T2] PARTIAL\n\n"
        "DECISION:\nRetry",
    ]
    if loops == 5:
        sections.append(
            "LOOP 3\n"
            f"SELECTED_TASK:\n{queue_extra} Split out the fragile part for targeted correction\n\n"
            "REASON_FOR_SELECTION:\nThe original queue was too linear. The fragile section now blocks confidence more than the broader plan does.\n\n"
            "EXECUTE:\nPulled the weak component into its own task, rolled back one overconfident assumption, and replaced it with a lower-risk alternative.\n\n"
            "RESULT:\nThe correction improved robustness, but it also made the final plan slightly less ambitious than the first draft.\n\n"
            "CHECK:\nThat tradeoff is acceptable. Reliability matters more than preserving the most optimistic scope.\n\n"
            "LEARN:\nWhen a subtask remains unstable, lowering ambition can be better than layering more cleverness onto a weak base.\n\n"
            "UPDATE_STATE:\nThe plan is narrower but more defensible.\n\n"
            "UPDATE_QUEUE:\n[T3] Test the plan against one failure scenario (priority: high)\n[T4] Package the output with explicit caveats (priority: medium)\n"
            f"{queue_extra} DONE\n\n"
            "DECISION:\nContinue"
        )
    test_loop_no = 4 if loops == 5 else 3
    final_loop_no = 5 if loops == 5 else 4
    sections.append(
        f"LOOP {test_loop_no}\n"
        "SELECTED_TASK:\n[T3] Test the plan against one failure scenario\n\n"
        "REASON_FOR_SELECTION:\nThe remaining question is not whether the plan is elegant, but whether it fails acceptably.\n\n"
        "EXECUTE:\nRan the draft mentally or operationally against one realistic bad case and checked what would still break.\n\n"
        "RESULT:\nThe revised plan survives the main scenario, but not perfectly. One residual risk remains documented because removing it now would cost too much time for the likely benefit.\n\n"
        "CHECK:\nThis is good enough for a controlled first rollout or handoff, not for claiming the issue is solved forever.\n\n"
        "LEARN:\nResidual risk is acceptable when it is visible, bounded, and cheaper to monitor than to eliminate immediately.\n\n"
        "UPDATE_STATE:\nThe output is usable, with one unresolved edge case left explicit.\n\n"
        "UPDATE_QUEUE:\n[T4] Package the output with explicit caveats (priority: high)\n[T3] DONE\n\n"
        "DECISION:\nContinue"
    )
    sections.append(
        f"LOOP {final_loop_no}\n"
        "SELECTED_TASK:\n[T4] Package the output with explicit caveats\n\n"
        "REASON_FOR_SELECTION:\nThe objective is complete only when the tradeoffs, caveats, and next steps are written clearly enough for execution.\n\n"
        "EXECUTE:\nCompiled the final recommendation, the correction history, the remaining edge case, and a short note on where the next review should focus.\n\n"
        "RESULT:\nThe output is ready for a first implementation or operational pass. It is intentionally not perfect, but it is much safer than the initial direction.\n\n"
        "CHECK:\nStopping is justified because "
        + stop_reason
        + ".\n\n"
        "LEARN:\nA strong agent should stop when the marginal value of further polishing drops below the cost of delay.\n\n"
        "UPDATE_STATE:\nObjective completed with residual uncertainty preserved instead of hidden.\n\n"
        "UPDATE_QUEUE:\nAll tasks complete.\n\n"
        "DECISION:\nStop\n\n"
        "STOP_REASON:\n"
        + stop_reason
        + ".\n\n"
        "FINAL_OUTPUT:\n"
        "Delivered a practical first-pass output with a corrected plan, a documented residual risk, a clear next review point, and caveats that make the result safe enough to use without pretending it is final."
    )
    return "\n\n".join(sections)


def rejected_text(category: str, objective: str, constraint: str, idx: int) -> str:
    return (
        "OBJECTIVE:\n"
        f"{objective.capitalize()} under the condition that {constraint}.\n\n"
        "CURRENT_STATE:\n"
        "The situation needs improvement, so the best approach is to make a reasonable plan and refine it later if necessary.\n\n"
        "TASK_QUEUE:\n"
        "[T1] Gather inputs\n"
        "[T2] Draft a plan\n"
        "[T3] Finalize the output\n\n"
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
        "FINAL_OUTPUT:\n"
        "Delivered a reasonable plan with concise notes and room for later tuning if needed."
    )


def main() -> None:
    random.seed(7)
    samples = []
    idx = 0
    for category, count in CATEGORIES:
        for n in range(count):
            objective = DOMAINS[category][n % len(DOMAINS[category])]
            constraint = CONSTRAINTS[(idx + n) % len(CONSTRAINTS)]
            samples.append(
                {
                    "prompt": build_prompt(category, objective, constraint),
                    "chosen": chosen_text(category, objective, constraint, idx),
                    "rejected": rejected_text(category, objective, constraint, idx),
                }
            )
            idx += 1

    assert len(samples) == TOTAL
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
