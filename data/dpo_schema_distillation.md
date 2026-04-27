Generate one DPO training sample for Versper-V1-Evo.

Return ONLY valid JSONL.

Schema:

{
"prompt": "Objective: <realistic task>",
"chosen": "<high quality autonomous iterative solution>",
"rejected": "<weaker static shallow solution>"
}

Rules:

Chosen should include:

* task decomposition
* loops
* learning
* state updates
* stopping decision
* practical final result

Rejected should contain one or more flaws:

* shallow planning
* no iteration
* fake completion
* no learning
* repetitive behavior
* vague output

Use realistic domains:
coding, research, finance, business, productivity.

Now generate ONE unique high-quality sample.



"Excellent, continue generating, and complete the generation of all 3,000 samples according to the category distribution:

Coding:        600
Finance:       300
Research:      300
Business:      200
Productivity:  100
Total:        1500"