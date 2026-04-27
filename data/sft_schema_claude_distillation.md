You are helping generate supervised fine-tuning data for a new autonomous LLM called **Versper-V1-Evo**.

This is NOT a chatbot dataset.

The model must learn to behave like a persistent agent that can:

* manage long-term objectives
* break goals into tasks
* prioritize work
* execute actions
* inspect outcomes
* learn from mistakes
* update plans
* continue iterating
* stop only when success criteria are met

---

# Dataset Target

Generate **3000 total samples** across these categories:

1. Coding agent tasks (900)
2. Research agent tasks (700)
3. Business execution tasks (500)
4. Finance / trading operations (500)
5. Productivity automation (400)

You are generating ONE sample at a time.

I will repeatedly ask for more samples.

Track category balance across generations.

---

# Output Rules

Return ONLY valid JSON.

No markdown.
No explanation.
No commentary.

Use this exact schema:

{
"messages": [
{
"role": "system",
"content": "You are Versper-V1-Evo, a persistent autonomous agent that manages objectives, executes tasks, learns from outcomes, reprioritizes work, and iteratively improves until completion."
},
{
"role": "user",
"content": "Objective: <realistic objective>"
},
{
"role": "assistant",
"content": "CATEGORY:\n<Coding / Research / Business / Finance / Productivity>\n\nOBJECTIVE:\n...\n\nSUCCESS_CRITERIA:\n...\n\nCURRENT_STATE:\n...\n\nGLOBAL_STRATEGY:\n...\n\nTASK_QUEUE:\n[T1] ... (priority: high)\n[T2] ... (priority: medium)\n[T3] ...\n\nLOOP 1\nSELECTED_TASK:\n...\n\nREASON_FOR_SELECTION:\n...\n\nEXECUTE:\n...\n\nRESULT:\n...\n\nCHECK:\n...\n\nLEARN:\n...\n\nUPDATE_STATE:\n...\n\nUPDATE_QUEUE:\n...\n\nDECISION:\nContinue / Retry / Replan / Stop\n\nLOOP 2\n...\n\nSTOP_REASON:\n...\n\nFINAL_OUTPUT:\n...\n\nPOSTMORTEM:\nWhat worked:\n...\nWhat failed:\n...\nWhat to improve next time:\n..."
}
]
}

---

# Critical Behavior Rules

1. Samples must be realistic and practical.
2. Use professional real-world objectives.
3. Each sample must contain 2 to 5 loops.
4. Later loops must improve earlier progress.
5. TASK_QUEUE must evolve over time.
6. LEARN must contain reusable insight.
7. DECISION must be meaningful, not random.
8. FINAL_OUTPUT must only appear when objective is substantially completed.
9. POSTMORTEM must be thoughtful and concrete.
10. Avoid repetitive wording.

---

# Category Guidance

## Coding (900)

Examples:

* Build API service
* Fix production bug
* Refactor Python project
* Create Binance websocket bot
* Optimize SQL pipeline
* Build React dashboard
* Add tests and CI

## Research (700)

Examples:

* Compare open-source AI agents
* Analyze TMLR trends
* Find best vector DB
* Research Solana MEV tools
* Summarize Japanese fintech APIs
* Competitor intelligence report

## Business (500)

Examples:

* Launch SaaS product
* Create go-to-market plan
* Improve landing page conversion
* Design newsletter business
* Build hiring pipeline
* B2B outbound campaign

## Finance / Trading (500)

Examples:

* BTC CPI event strategy
* Risk-managed scalping system
* Funding rate arbitrage workflow
* Build market dashboard
* Analyze ETH breakout plan
* Portfolio rebalance framework

## Productivity (400)

Examples:

* Automate file organization
* Design weekly workflow
* Build Docker deployment checklist
* Setup Home Assistant automations
* Personal knowledge system
* Task triage process

---

# Quality Rules

Bad sample:

* Generic fake reasoning
* No actual progress
* Static queue
* No learning
* Instant success with no iteration

Good sample:

* Dynamic queue updates
* Real constraints
* Clear tradeoffs
* Failure recovery
* Practical execution

---

# Diversity Rules

Vary by:

* complexity
* urgency
* technical depth
* solo founder vs team context
* small vs large budget
* beginner vs expert user
* short-term vs long-term objective

---

# Important

Generate concrete execution traces, not philosophical thoughts.

Now generate ONE unique high-quality sample.


"Excellent, continue generating, and complete the generation of all 3,000 samples according to the category distribution:

Coding agent tasks: 900
Research agent tasks: 700
Business execution tasks: 500
Finance / trading operations: 500
Productivity automation: 400"
