import json
import random
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "You are Versper-V1-Evo, a persistent autonomous agent that manages "
    "objectives, executes tasks, learns from outcomes, reprioritizes work, "
    "and iteratively improves until completion."
)

TARGETS = {
    "Coding": 900,
    "Research": 700,
    "Business": 500,
    "Finance": 500,
    "Productivity": 400,
}

CATEGORY_LABELS = {
    "Coding": "Coding",
    "Research": "Research",
    "Business": "Business",
    "Finance": "Finance",
    "Productivity": "Productivity",
}

SECTORS = [
    "fintech",
    "healthtech",
    "logistics",
    "e-commerce",
    "developer tools",
    "climate tech",
    "media analytics",
    "edtech",
    "cybersecurity",
    "B2B SaaS",
]

TECH_STACKS = [
    "Python + FastAPI + PostgreSQL",
    "TypeScript + Node.js + Redis",
    "Go + ClickHouse + Kafka",
    "Python + Airflow + BigQuery",
    "React + Next.js + Supabase",
    "Rust + PostgreSQL + gRPC",
    "Java + Spring Boot + MySQL",
]

RESEARCH_DOMAINS = [
    "open-source coding agents",
    "European payment APIs",
    "vector database vendors",
    "AI observability tooling",
    "developer-first analytics products",
    "SMB procurement platforms",
    "Solana infrastructure providers",
    "enterprise note-taking software",
]

BUSINESS_MOTIONS = [
    "launch a vertical SaaS offer",
    "improve outbound conversion",
    "build a partnership pipeline",
    "raise demo-to-trial conversion",
    "design a customer onboarding program",
    "rework pricing and packaging",
    "stand up an affiliate channel",
]

FINANCE_PLAYS = [
    "BTC CPI-event trading framework",
    "ETH swing trading plan",
    "market-neutral funding rate capture",
    "portfolio rebalance system",
    "small-cap crypto risk screen",
    "intraday futures execution checklist",
    "treasury cash allocation workflow",
]

PRODUCTIVITY_SYSTEMS = [
    "weekly operations review system",
    "file intake and tagging automation",
    "meeting notes knowledge base workflow",
    "task triage and escalation process",
    "home lab maintenance checklist",
    "sales inbox processing automation",
    "personal research capture pipeline",
]

CONSTRAINTS = [
    "budget is capped this quarter",
    "the team is small and time-constrained",
    "the deadline is tied to a customer commitment",
    "the current process is unreliable under load",
    "stakeholders need auditable outputs",
    "the owner cannot pause existing operations",
    "tooling is fragmented across multiple systems",
]

ROLE_CONTEXTS = [
    "solo founder",
    "staff engineer",
    "operations manager",
    "product lead",
    "research analyst",
    "growth operator",
    "portfolio manager",
]

PRIORITY_WORDS = ["high", "medium", "low"]


def slug(text: str) -> str:
    return (
        text.lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )


def count_existing(path: Path) -> Counter:
    counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            content = obj["messages"][2]["content"]
            counts[content.split("\n", 2)[1].strip()] += 1
    return counts


def objective_for(category: str, idx: int, rnd: random.Random) -> tuple[str, dict]:
    sector = SECTORS[idx % len(SECTORS)]
    role = ROLE_CONTEXTS[(idx * 3) % len(ROLE_CONTEXTS)]
    constraint = CONSTRAINTS[(idx * 5) % len(CONSTRAINTS)]
    if category == "Coding":
        stack = TECH_STACKS[idx % len(TECH_STACKS)]
        objective = (
            f"Build or improve a production workflow for a {sector} team using "
            f"{stack}, where the {role} needs a reliable delivery path and {constraint}."
        )
        meta = {
            "deliverable": rnd.choice(
                [
                    "event ingestion service",
                    "internal admin dashboard",
                    "background job scheduler",
                    "search API",
                    "customer-facing reporting endpoint",
                    "CI hardening project",
                    "test coverage and release automation",
                ]
            ),
            "metric": rnd.choice(
                [
                    "error rate below 0.5%",
                    "p95 latency under 250ms",
                    "coverage above 80%",
                    "zero duplicate records after retries",
                    "green CI on main branch",
                    "deployment rollback path documented",
                ]
            ),
        }
        return objective, meta
    if category == "Research":
        topic = RESEARCH_DOMAINS[idx % len(RESEARCH_DOMAINS)]
        objective = (
            f"Research {topic} for a {sector} company. The {role} needs a practical "
            f"recommendation memo and {constraint}."
        )
        meta = {
            "deliverable": rnd.choice(
                [
                    "decision memo",
                    "vendor comparison",
                    "market landscape brief",
                    "shortlist with scoring model",
                    "competitive intelligence report",
                ]
            ),
            "metric": rnd.choice(
                [
                    "at least three strong candidates with tradeoffs",
                    "clear recommendation with risks",
                    "source-backed comparison matrix",
                    "decision ready for executive review",
                ]
            ),
        }
        return objective, meta
    if category == "Business":
        motion = BUSINESS_MOTIONS[idx % len(BUSINESS_MOTIONS)]
        objective = (
            f"Create an execution plan to {motion} for a {sector} business. The "
            f"{role} needs something operational, not a slide deck, and {constraint}."
        )
        meta = {
            "deliverable": rnd.choice(
                [
                    "90-day rollout plan",
                    "funnel improvement playbook",
                    "pipeline build system",
                    "pricing experiment roadmap",
                    "weekly operating cadence",
                ]
            ),
            "metric": rnd.choice(
                [
                    "owners assigned for every workstream",
                    "first experiment live this month",
                    "leading indicators defined",
                    "execution blockers surfaced early",
                ]
            ),
        }
        return objective, meta
    if category == "Finance":
        play = FINANCE_PLAYS[idx % len(FINANCE_PLAYS)]
        objective = (
            f"Design and validate a {play} for a {role} managing capital in a "
            f"{sector} context, where {constraint}."
        )
        meta = {
            "deliverable": rnd.choice(
                [
                    "risk-managed playbook",
                    "execution checklist",
                    "position sizing framework",
                    "dashboard specification",
                    "scenario response plan",
                ]
            ),
            "metric": rnd.choice(
                [
                    "max drawdown constraint defined",
                    "entry and exit conditions explicit",
                    "risk limits enforceable before execution",
                    "manual review gates documented",
                ]
            ),
        }
        return objective, meta
    system = PRODUCTIVITY_SYSTEMS[idx % len(PRODUCTIVITY_SYSTEMS)]
    objective = (
        f"Build a {system} for a {role} in {sector}. It must reduce manual overhead and "
        f"{constraint}."
    )
    meta = {
        "deliverable": rnd.choice(
            [
                "automation runbook",
                "repeatable checklist",
                "folder and tagging convention",
                "calendar and review workflow",
                "triage SOP",
            ]
        ),
        "metric": rnd.choice(
            [
                "weekly maintenance under 30 minutes",
                "fewer dropped tasks",
                "clear exception handling",
                "handoff steps documented",
            ]
        ),
    }
    return objective, meta


def queue_items(category: str, meta: dict, idx: int) -> list[str]:
    common = [
        "Audit current state and confirm constraints",
        "Define measurable success criteria and failure cases",
    ]
    category_specific = {
        "Coding": [
            f"Design architecture for the {meta['deliverable']}",
            "Implement the highest-risk path first",
            "Add tests and verification steps",
            "Harden deployment and rollback process",
        ],
        "Research": [
            "Collect primary sources and recent evidence",
            "Build a comparison matrix with weighted criteria",
            "Interview or infer likely operator constraints",
            "Draft recommendation and open risks",
        ],
        "Business": [
            "Map funnel or process bottlenecks",
            "Rank initiatives by effort and expected impact",
            "Assign owners and deadlines",
            "Define first operating review cadence",
        ],
        "Finance": [
            "Define risk envelope before alpha logic",
            "Build scenarios for adverse market moves",
            "Specify entries, exits, and invalidation rules",
            "Create monitoring and exposure limits",
        ],
        "Productivity": [
            "Document current workflow pain points",
            "Design the automation or checklist backbone",
            "Test edge cases and exception paths",
            "Train the owner on the new routine",
        ],
    }[category]
    items = common + category_specific
    rnd = random.Random(idx * 17 + len(category))
    rnd.shuffle(items)
    return items


def make_loop_block(loop_no: int, total_loops: int, queue: list[tuple[str, str]], category: str, idx: int) -> tuple[str, str]:
    task_id, task_text = queue.pop(0)
    reason_templates = {
        "Coding": "This task removes the main implementation risk and unlocks dependent work.",
        "Research": "Without this evidence, any recommendation would be weak or speculative.",
        "Business": "This task directly affects execution quality and prevents downstream churn.",
        "Finance": "Risk must be bounded before the plan can be trusted or executed.",
        "Productivity": "Fixing the highest-friction step first creates immediate leverage for the rest of the system.",
    }
    exec_templates = {
        "Coding": [
            "Reviewed the codebase boundary, sketched interfaces, and implemented the first version with minimal abstractions.",
            "Added a narrow proof of correctness with a local test harness and instrumented error paths.",
            "Tightened the workflow around the failure mode found in the prior loop and removed unnecessary branching.",
        ],
        "Research": [
            "Collected vendor docs, analyst notes, and user feedback, then normalized claims into a comparison sheet.",
            "Separated marketing claims from operator-level evidence and flagged unsupported statements.",
            "Re-ranked options after checking integration friction, pricing, and migration risk.",
        ],
        "Business": [
            "Mapped the current funnel, quantified the biggest drop-off, and converted vague ideas into owner-based tasks.",
            "Interviewed internal stakeholders, aligned on dependencies, and cut low-leverage initiatives.",
            "Turned strategy notes into a weekly execution rhythm with explicit KPIs and review points.",
        ],
        "Finance": [
            "Defined the market scenario, wrote down guardrails, and back-checked the plan against recent volatility.",
            "Stress-tested the setup under an adverse move and reduced position sizing where fragility was obvious.",
            "Translated the idea into an execution checklist with pre-trade and post-trade controls.",
        ],
        "Productivity": [
            "Observed the current workflow, removed redundant steps, and proposed a simpler default path.",
            "Configured naming, routing, or review rules and tested them against realistic edge cases.",
            "Added an exception lane so the system fails gracefully instead of silently dropping work.",
        ],
    }
    result_templates = {
        "Coding": [
            "The core path worked, but one dependency assumption failed under realistic input volume.",
            "The implementation passed the first check, though one integration boundary remained brittle.",
            "The revised path reduced failure risk and made the next task cheaper to validate.",
        ],
        "Research": [
            "Two candidates looked strong initially, but one fell behind once switching cost and data portability were considered.",
            "The evidence base improved and removed one false lead that had attractive marketing but weak operational proof.",
            "The shortlist became clearer after separating must-haves from nice-to-haves.",
        ],
        "Business": [
            "The highest-impact bottleneck was narrower than expected, which allowed the plan to focus on fewer changes.",
            "One workstream was blocked by data quality, so the sequence had to be adjusted before scaling outreach.",
            "The refined plan now has a credible owner path and fewer hidden dependencies.",
        ],
        "Finance": [
            "The opportunity remained viable, but the original size assumption was too aggressive for the observed volatility.",
            "One scenario produced asymmetric downside, so the execution rules were tightened before approval.",
            "The updated framework now reflects actual liquidity and operational constraints.",
        ],
        "Productivity": [
            "The simplified routine saved time immediately, but one exception path still created manual cleanup.",
            "The automation covered the default case well and exposed one naming inconsistency that needed correction.",
            "The updated system became easier to maintain because the review step now catches drift early.",
        ],
    }
    check_templates = {
        "Coding": "Verified behavior against the stated metric and checked whether the change reduced implementation risk.",
        "Research": "Checked source quality, evidence recency, and whether the recommendation remained grounded in facts.",
        "Business": "Checked owner clarity, milestone realism, and whether the task improved execution readiness.",
        "Finance": "Checked drawdown assumptions, liquidity realism, and whether the plan stayed within limits.",
        "Productivity": "Checked whether the new workflow actually reduced friction and whether exceptions were controlled.",
    }
    learn_templates = {
        "Coding": "Reusable insight: implement the riskiest integration before polishing interfaces; it prevents elegant but wrong architecture.",
        "Research": "Reusable insight: vendor claims are not evidence; operator constraints and switching cost change rankings materially.",
        "Business": "Reusable insight: execution plans fail when ownership is fuzzy; every initiative needs one accountable driver and one metric.",
        "Finance": "Reusable insight: sizing discipline matters more than idea quality when market conditions change faster than analysis.",
        "Productivity": "Reusable insight: the best automation is not the most elaborate one; it is the one that survives edge cases with low maintenance.",
    }
    update_state_templates = {
        "Coding": "The technical plan is clearer, one risky assumption has been corrected, and the implementation path is narrower.",
        "Research": "The evidence base is stronger, the option set is smaller, and recommendation confidence has improved.",
        "Business": "The operating plan is more concrete, dependencies are visible, and wasted motion has been reduced.",
        "Finance": "The trading or portfolio framework is better bounded, with clearer triggers and tighter controls.",
        "Productivity": "The workflow is more reliable, exception handling is explicit, and the owner has a clearer routine.",
    }
    decision = "Continue" if loop_no < total_loops else "Stop"
    if loop_no == total_loops - 1 and total_loops > 2:
        decision = "Replan"
    if loop_no == total_loops:
        decision = "Stop"

    execute = exec_templates[category][(idx + loop_no) % len(exec_templates[category])]
    result = result_templates[category][(idx * 2 + loop_no) % len(result_templates[category])]
    if loop_no == total_loops:
        result = "The final pass resolved the remaining blocker and brought the objective substantially to completion."

    if loop_no < total_loops:
        next_task_text = queue[0][1] if queue else "Close the remaining open items"
        update_queue = f"{task_id} DONE\n{queue[0][0]} {next_task_text} (priority: high)" if queue else f"{task_id} DONE\nQueue nearly empty."
    else:
        update_queue = "All planned tasks complete. Queue empty."

    block = (
        f"LOOP {loop_no}\n"
        f"SELECTED_TASK:\n{task_id} {task_text}\n\n"
        f"REASON_FOR_SELECTION:\n{reason_templates[category]}\n\n"
        f"EXECUTE:\n{execute}\n\n"
        f"RESULT:\n{result}\n\n"
        f"CHECK:\n{check_templates[category]}\n\n"
        f"LEARN:\n{learn_templates[category]}\n\n"
        f"UPDATE_STATE:\n{update_state_templates[category]}\n\n"
        f"UPDATE_QUEUE:\n{update_queue}\n\n"
        f"DECISION:\n{decision}"
    )
    return block, decision


def build_sample(category: str, idx: int) -> dict:
    rnd = random.Random(f"{category}-{idx}")
    objective, meta = objective_for(category, idx, rnd)
    loops = 2 + (idx % 4)
    raw_queue = queue_items(category, meta, idx)
    queue = []
    for pos, item in enumerate(raw_queue[: loops + 3], start=1):
        priority = PRIORITY_WORDS[min(pos - 1, len(PRIORITY_WORDS) - 1)]
        queue.append((f"[T{pos}]", f"{item} (priority: {priority})"))

    success_criteria = [
        f"- Deliver the planned {meta['deliverable']} with concrete progress, not just analysis",
        f"- Show iterative improvement across {loops} loops",
        f"- End with {meta['metric']}",
        "- Leave a clear final output and a concrete postmortem",
    ]
    current_state = (
        f"Objective is active for a {rnd.choice(ROLE_CONTEXTS)} in {SECTORS[(idx + 2) % len(SECTORS)]}. "
        "Some context exists, but the execution path is incomplete and the queue needs active reprioritization."
    )
    global_strategy = (
        "1. Establish the real constraint before taking broad action\n"
        "2. Tackle the highest-risk or highest-uncertainty task first\n"
        "3. Use each loop to narrow the plan based on observed results\n"
        "4. Stop only when the outcome is concrete enough to hand off or execute"
    )
    initial_queue = "\n".join(f"{task_id} {task_text}" for task_id, task_text in queue)

    queue_for_loops = [(task_id, task_text.rsplit(" (priority:", 1)[0]) for task_id, task_text in queue]
    loop_blocks = []
    last_decision = "Continue"
    for loop_no in range(1, loops + 1):
        block, last_decision = make_loop_block(loop_no, loops, queue_for_loops, category, idx)
        loop_blocks.append(block)

    stop_reason = (
        "The objective is substantially completed, the remaining risks are documented, and no critical open task blocks handoff."
    )
    final_output = {
        "Coding": "A working implementation path, verification results, remaining risks, and clear next deployment steps.",
        "Research": "A decision-ready recommendation memo with shortlist, tradeoffs, and cited risk areas.",
        "Business": "An execution plan with owners, milestones, first experiments, and operating review cadence.",
        "Finance": "A risk-bounded playbook with scenarios, exposure limits, and explicit execution rules.",
        "Productivity": "A repeatable system with defaults, exception handling, and maintenance guidance.",
    }[category]
    postmortem = {
        "worked": {
            "Coding": "Working on the riskiest path early reduced rework and exposed integration issues before polish.",
            "Research": "Evidence-based ranking prevented attractive but weak options from surviving too long.",
            "Business": "Owner-based planning converted generic strategy into something the team can actually run.",
            "Finance": "Risk-first sequencing prevented the plan from drifting into uncontrolled exposure.",
            "Productivity": "Simplifying the default path produced immediate time savings and clearer maintenance behavior.",
        }[category],
        "failed": {
            "Coding": "The first implementation assumption was too optimistic and had to be tightened after validation.",
            "Research": "One early lead relied too heavily on vendor framing and not enough on operator friction.",
            "Business": "Some initial initiatives were too broad and created avoidable sequencing noise.",
            "Finance": "The original sizing or timing assumption understated downside in one realistic scenario.",
            "Productivity": "One exception path remained manual longer than expected and needed a dedicated fix.",
        }[category],
        "improve": {
            "Coding": "Next time, build lightweight verification earlier for the most failure-prone boundary.",
            "Research": "Next time, establish the evaluation rubric before collecting too many sources.",
            "Business": "Next time, align on leading metrics before assigning broader workstreams.",
            "Finance": "Next time, codify the invalidation rule before discussing upside scenarios.",
            "Productivity": "Next time, model exception handling earlier instead of assuming the default path is enough.",
        }[category],
    }

    assistant_content = (
        f"CATEGORY:\n{CATEGORY_LABELS[category]}\n\n"
        f"OBJECTIVE:\n{objective}\n\n"
        f"SUCCESS_CRITERIA:\n" + "\n".join(success_criteria) + "\n\n"
        f"CURRENT_STATE:\n{current_state}\n\n"
        f"GLOBAL_STRATEGY:\n{global_strategy}\n\n"
        f"TASK_QUEUE:\n{initial_queue}\n\n"
        + "\n\n".join(loop_blocks)
        + f"\n\nSTOP_REASON:\n{stop_reason}\n\n"
        f"FINAL_OUTPUT:\n{final_output}\n\n"
        "POSTMORTEM:\n"
        f"What worked:\n{postmortem['worked']}\n"
        f"What failed:\n{postmortem['failed']}\n"
        f"What to improve next time:\n{postmortem['improve']}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Objective: {objective}"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    path = Path("data/sft.jsonl")
    existing = count_existing(path)
    needed = {category: TARGETS[category] - existing.get(category, 0) for category in TARGETS}
    total_needed = sum(needed.values())
    if total_needed <= 0:
        print("No records needed.")
        return

    with path.open("a", encoding="utf-8") as handle:
        for category, count in needed.items():
            start_idx = existing.get(category, 0)
            for offset in range(count):
                sample = build_sample(category, start_idx + offset)
                handle.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")) + "\n")

    final_counts = count_existing(path)
    print("Final counts:", dict(final_counts))
    print("Final total:", sum(final_counts.values()))


if __name__ == "__main__":
    main()
