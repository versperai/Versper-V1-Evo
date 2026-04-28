from __future__ import annotations

from dataclasses import dataclass


ARCHETYPE_CYCLE = (
    "simplified_system_design",
    "simplified_system_design",
    "simplified_system_design",
    "simplified_system_design",
    "missing_key_modules",
    "missing_key_modules",
    "missing_key_modules",
    "partially_wrong_reasoning",
    "partially_wrong_reasoning",
    "weak_generic_answer",
)

OPENERS = [
    "Start with the smallest workable version and avoid adding extra machinery too early.",
    "Use a narrow first pass so the operator can move quickly without waiting for a perfect design.",
    "Keep the plan simple enough to execute immediately and defer deeper optimization.",
]

WRONG_ASSUMPTIONS = [
    "the main bottleneck is tooling complexity rather than the operating constraint",
    "most risks can be tolerated until after the first rollout",
    "one good draft is usually enough if the core idea looks reasonable",
]

MISSING_MODULES = [
    "rollback handling",
    "explicit constraint checks",
    "failure monitoring",
    "handoff criteria",
]


@dataclass(frozen=True)
class Candidate:
    text: str
    correctness: float
    completeness: float
    structure: float
    archetype: str
    label: str

    @property
    def score(self) -> float:
        return 0.6 * self.correctness + 0.3 * self.completeness + 0.1 * self.structure


def select_archetype(idx: int) -> str:
    return ARCHETYPE_CYCLE[idx % len(ARCHETYPE_CYCLE)]


def build_candidate_pool(objective: str, constraint: str, idx: int) -> list[Candidate]:
    archetype = select_archetype(idx)
    medium = _build_medium_candidate(objective, constraint, idx, archetype)
    fallback = _build_weak_candidate(objective, constraint, idx)
    alternate = _build_alternate_candidate(objective, constraint, idx, archetype)
    return [medium, fallback, alternate]


def choose_rejected(objective: str, constraint: str, idx: int) -> str:
    pool = build_candidate_pool(objective, constraint, idx)
    target_archetype = select_archetype(idx)
    typed_pool = [candidate for candidate in pool if candidate.archetype == target_archetype]
    ranked_pool = typed_pool or pool
    return max(ranked_pool, key=lambda candidate: candidate.score).text


def _build_medium_candidate(objective: str, constraint: str, idx: int, archetype: str) -> Candidate:
    if archetype == "simplified_system_design":
        text = (
            "OBJECTIVE:\n"
            f"{objective.capitalize()} while keeping the process lightweight under the condition that {constraint}.\n\n"
            "CURRENT_STATE:\n"
            f"{OPENERS[idx % len(OPENERS)]}\n\n"
            "TASK_QUEUE:\n"
            "[T1] Outline the direct solution path\n"
            "[T2] Pick the lowest-friction implementation approach\n"
            "[T3] Summarize the rollout steps\n\n"
            "LOOP 1\n"
            "SELECTED_TASK:\n[T1] Outline the direct solution path\n\n"
            "EXECUTE:\nMapped the main objective to a short delivery sequence and ignored deeper branching for now.\n\n"
            "RESULT:\nThe high-level path is clear enough to proceed without extra analysis.\n\n"
            "UPDATE_STATE:\nA simple plan exists.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 2\n"
            "SELECTED_TASK:\n[T2] Pick the lowest-friction implementation approach\n\n"
            "EXECUTE:\nChose the most straightforward design and removed heavier safeguards to keep execution fast.\n\n"
            "RESULT:\nThe approach should work if normal conditions hold, but it does not cover unusual failure cases.\n\n"
            "LEARN:\nA direct design is usually more practical than a fully instrumented one.\n\n"
            "UPDATE_STATE:\nThe implementation direction is set.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 3\n"
            "SELECTED_TASK:\n[T3] Summarize the rollout steps\n\n"
            "EXECUTE:\nWrote a concise rollout sequence and noted that deeper validation can happen after initial execution.\n\n"
            "RESULT:\nThe output is usable, but some safeguards and feedback loops are intentionally deferred.\n\n"
            "DECISION:\nStop\n\n"
            "FINAL_OUTPUT:\nDelivered a simplified design with clear execution steps, but without detailed recovery or recursive verification."
        )
        return Candidate(text, 0.74, 0.63, 0.84, archetype, "medium")

    if archetype == "missing_key_modules":
        missing = MISSING_MODULES[idx % len(MISSING_MODULES)]
        text = (
            "OBJECTIVE:\n"
            f"{objective.capitalize()} while working under the condition that {constraint}.\n\n"
            "CURRENT_STATE:\n"
            "A structured answer is needed, but the fastest path is to cover the main workflow first and treat support pieces as optional.\n\n"
            "TASK_QUEUE:\n"
            "[T1] Confirm the main path\n"
            "[T2] Define the operating plan\n"
            "[T3] Prepare the handoff summary\n\n"
            "LOOP 1\n"
            "SELECTED_TASK:\n[T1] Confirm the main path\n\n"
            "EXECUTE:\nChecked the core objective and identified the shortest implementation route.\n\n"
            "RESULT:\nThe main path looks viable.\n\n"
            "UPDATE_STATE:\nThe primary approach is clear.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 2\n"
            "SELECTED_TASK:\n[T2] Define the operating plan\n\n"
            "EXECUTE:\nBuilt a plan with sequencing, ownership, and expected outputs, but skipped "
            f"{missing} to avoid slowing the first pass.\n\n"
            "RESULT:\nThe plan covers the main workflow, though one support module is absent.\n\n"
            "LEARN:\nThe core path matters more than completing every control layer up front.\n\n"
            "UPDATE_STATE:\nThe plan is mostly complete.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 3\n"
            "SELECTED_TASK:\n[T3] Prepare the handoff summary\n\n"
            "EXECUTE:\nSummarized the rollout and highlighted the expected next action.\n\n"
            "RESULT:\nThe output is organized, but it still relies on an unstated support mechanism.\n\n"
            "DECISION:\nStop\n\n"
            "FINAL_OUTPUT:\nDelivered a structured plan with the main modules in place, but one important supporting component is missing."
        )
        return Candidate(text, 0.7, 0.68, 0.83, archetype, "medium")

    if archetype == "partially_wrong_reasoning":
        wrong = WRONG_ASSUMPTIONS[idx % len(WRONG_ASSUMPTIONS)]
        text = (
            "OBJECTIVE:\n"
            f"{objective.capitalize()} while responding under the condition that {constraint}.\n\n"
            "CURRENT_STATE:\n"
            "The problem appears manageable if the plan stays focused on the most visible blocker.\n\n"
            "TASK_QUEUE:\n"
            "[T1] Identify the main blocker\n"
            "[T2] Draft the response around that blocker\n"
            "[T3] Finalize the recommendation\n\n"
            "LOOP 1\n"
            "SELECTED_TASK:\n[T1] Identify the main blocker\n\n"
            "EXECUTE:\nAssumed that "
            f"{wrong}.\n\n"
            "RESULT:\nThat assumption gives a coherent direction, so the rest of the plan is built around it.\n\n"
            "UPDATE_STATE:\nA single-driver explanation now anchors the plan.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 2\n"
            "SELECTED_TASK:\n[T2] Draft the response around that blocker\n\n"
            "EXECUTE:\nBuilt a plan that solves the assumed blocker cleanly and keeps extra checks minimal.\n\n"
            "RESULT:\nThe answer is internally consistent, but it may underweight constraints that do not fit the initial assumption.\n\n"
            "LEARN:\nA focused explanation is easier to execute than a multi-branch plan.\n\n"
            "UPDATE_STATE:\nThe draft is coherent but narrow.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 3\n"
            "SELECTED_TASK:\n[T3] Finalize the recommendation\n\n"
            "EXECUTE:\nPolished the recommendation and kept only brief caveats.\n\n"
            "RESULT:\nThe output looks convincing, though it depends on an assumption that was not pressure-tested.\n\n"
            "DECISION:\nStop\n\n"
            "FINAL_OUTPUT:\nDelivered a plausible plan that addresses one dominant explanation well, but may be incomplete if the initial reasoning is off."
        )
        return Candidate(text, 0.62, 0.58, 0.82, archetype, "medium")

    return _build_weak_candidate(objective, constraint, idx)


def _build_weak_candidate(objective: str, constraint: str, idx: int) -> Candidate:
    text = (
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while keeping effort low under the condition that {constraint}.\n\n"
        "CURRENT_STATE:\n"
        "The task mainly needs a reasonable plan, so a standard approach should be enough.\n\n"
        "TASK_QUEUE:\n"
        "[T1] Gather the obvious inputs\n"
        "[T2] Draft a simple plan\n"
        "[T3] Deliver the recommendation\n\n"
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Gather the obvious inputs\n\n"
        "EXECUTE:\nCollected the most visible requirements and skipped deeper checks.\n\n"
        "RESULT:\nThe basics look sufficient to continue.\n\n"
        "UPDATE_STATE:\nEnough context exists to draft a response.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Draft a simple plan\n\n"
        "EXECUTE:\nOutlined a direct plan with broad steps and limited detail.\n\n"
        "RESULT:\nThe plan is understandable, though several implementation details remain implicit.\n\n"
        "LEARN:\nA concise answer is often enough for an initial pass.\n\n"
        "UPDATE_STATE:\nThe draft is ready.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 3\n"
        "SELECTED_TASK:\n[T3] Deliver the recommendation\n\n"
        "EXECUTE:\nFinalized the answer with a short summary and generic next steps.\n\n"
        "RESULT:\nThe output is usable as a starting point, but it does not deeply handle constraints, validation, or recovery.\n\n"
        "DECISION:\nStop\n\n"
        "FINAL_OUTPUT:\nDelivered a generic first-pass recommendation with a simple structure and room for later refinement."
    )
    return Candidate(text, 0.5, 0.42, 0.78, "weak_generic_answer", "weak")


def _build_alternate_candidate(objective: str, constraint: str, idx: int, archetype: str) -> Candidate:
    if archetype == "simplified_system_design":
        text = (
            "OBJECTIVE:\n"
            f"{objective.capitalize()} while working around the condition that {constraint}.\n\n"
            "CURRENT_STATE:\n"
            "The task can likely be solved by choosing one direct path and avoiding extra coordination loops.\n\n"
            "TASK_QUEUE:\n"
            "[T1] Pick the main execution path\n"
            "[T2] Describe the rollout\n"
            "[T3] Close with a brief recommendation\n\n"
            "LOOP 1\n"
            "SELECTED_TASK:\n[T1] Pick the main execution path\n\n"
            "EXECUTE:\nSelected the simplest architecture that appears to satisfy the objective.\n\n"
            "RESULT:\nThe path is coherent, but it assumes the constraint will not force major rework.\n\n"
            "UPDATE_STATE:\nThe architecture is tentatively chosen.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 2\n"
            "SELECTED_TASK:\n[T2] Describe the rollout\n\n"
            "EXECUTE:\nListed the core steps and left validation as a follow-up activity.\n\n"
            "RESULT:\nThe rollout is clear, but it lacks recovery and measurement detail.\n\n"
            "UPDATE_STATE:\nThe rollout outline is complete.\n\n"
            "DECISION:\nContinue\n\n"
            "LOOP 3\n"
            "SELECTED_TASK:\n[T3] Close with a brief recommendation\n\n"
            "EXECUTE:\nSummarized the path in a compact handoff.\n\n"
            "RESULT:\nThe answer is practical, though shallow in control design.\n\n"
            "DECISION:\nStop\n\n"
            "FINAL_OUTPUT:\nDelivered a compact execution path that favors simplicity over robustness."
        )
        return Candidate(text, 0.68, 0.56, 0.79, "simplified_system_design", "alternate")

    text = (
        "OBJECTIVE:\n"
        f"{objective.capitalize()} while staying within the condition that {constraint}.\n\n"
        "CURRENT_STATE:\n"
        "The problem can be addressed with a clear plan, even if some supporting details are deferred.\n\n"
        "TASK_QUEUE:\n"
        "[T1] Identify the main route\n"
        "[T2] Build the plan around it\n"
        "[T3] Finalize the summary\n\n"
        "LOOP 1\n"
        "SELECTED_TASK:\n[T1] Identify the main route\n\n"
        "EXECUTE:\nChose one plausible route and treated it as sufficient for the first answer.\n\n"
        "RESULT:\nThe route looks serviceable.\n\n"
        "UPDATE_STATE:\nA direction exists.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 2\n"
        "SELECTED_TASK:\n[T2] Build the plan around it\n\n"
        "EXECUTE:\nExpanded the route into a few execution steps but kept caveats light.\n\n"
        "RESULT:\nThe plan is readable, though one assumption still carries too much weight.\n\n"
        "UPDATE_STATE:\nThe draft is mostly ready.\n\n"
        "DECISION:\nContinue\n\n"
        "LOOP 3\n"
        "SELECTED_TASK:\n[T3] Finalize the summary\n\n"
        "EXECUTE:\nPrepared the final recommendation with a short conclusion.\n\n"
        "RESULT:\nThe output is plausible, but it does not fully defend its weakest assumption.\n\n"
        "DECISION:\nStop\n\n"
        "FINAL_OUTPUT:\nDelivered a plausible but lightly defended plan with limited validation depth."
    )
    return Candidate(text, 0.6, 0.52, 0.77, "partially_wrong_reasoning", "alternate")
