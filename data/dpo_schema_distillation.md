Generate one DPO training sample for Versper-V1-Evo.

Return ONLY valid JSONL.

Schema:

{
"prompt": "Objective: <realistic task>",
"chosen": "<strong but imperfect autonomous iterative solution>",
"rejected": "<weaker medium-grade or shallow solution>"
}

Core principle:

Do NOT produce perfect trajectories.

Versper-V1-Evo should learn realistic long-horizon agent behavior, not idealized flawless planning.

Chosen requirements:

* Must include task decomposition
* Must include 3 to 6 loops
* Must include at least 1 mistake, uncertainty, or bad assumption
* Must include at least 1 partial failure, correction, retry, rollback, or reprioritization
* Must include state updates
* Must include a stopping decision
* Must include a practical final result
* Must include residual uncertainty or unresolved risk
* Must include cost-aware or time-aware stopping when appropriate
* Must show that learning changes later behavior, but do not make every loop perfectly rational

Important chosen constraints:

* Do not make every loop a clean improvement
* Do not fully solve every problem
* Allow planning instability
* Allow queue reordering, task splitting, or newly spawned tasks
* Allow the agent to stop because the result is good enough under constraints, not because everything is perfect
* Residual uncertainty should remain visible in some samples
* Learning can be messy, tentative, or partial

Rejected requirements:

Rejected should NOT be a cartoonishly bad answer.

Rejected should be plausible but weaker.

It may include some structure or multiple steps, but should contain one or more flaws such as:

* weak or shallow planning
* linear execution with little adaptation
* fake completion
* insufficient learning
* failure to update priorities meaningfully
* overconfidence
* vague stopping logic
* no residual-risk handling

Preference target:

The chosen answer should be better because it shows stronger autonomy, better correction behavior, better uncertainty handling, and more defensible stopping decisions.

Do not make the rejected answer obviously incompetent.

Use realistic domains:
coding, research, finance, business, productivity.

Diversity requirements:

Vary:

* complexity
* urgency
* budget or time pressure
* solo operator vs team context
* type of mistake or uncertainty
* type of stopping reason
* degree of residual risk

Now generate ONE unique high-quality DPO sample for an evolution-capable autonomous agent.



"Excellent, continue generating, and complete the generation of all samples according to the category distribution:

Clean Agent DPO:
Coding:        600
Finance:       300
Research:      300
Business:      200
Productivity:  100

Chaos Agent DPO:
Coding:        200
Finance:       120
Research:      120
Business:       80
Productivity:   80

Total:        2100"


数据量够了，但是你用 脚本 排列组合填充 dpo.jsonl  dpo_chaos.jsonl  sft.jsonl 里的数据 会导致 太集中不符合自然 数据的正态分布，而且 质量也不如你 思考生成出来的数据质量高, 你一条条数据 提高质量
