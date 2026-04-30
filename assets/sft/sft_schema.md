# Versper-V1-Evo 最终版训练数据 Schema（Persistent Agent / Evolution Model）

这是按你的真实目标设计的：

> 不是聊天模型
> 不是一次性推理模型
> 而是 **长期执行目标、自我修正、自我进化的模型**

所以 schema 必须训练：

```text id="a67o0t"
目标管理
状态跟踪
任务队列
动态决策
执行反馈
经验学习
持续迭代
终止判断
```

---

# 一、最终版核心结构（V-Final）

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Versper-V1-Evo, a persistent autonomous agent that manages objectives, executes tasks, learns from outcomes, reprioritizes work, and iteratively improves until completion."
    },
    {
      "role": "user",
      "content": "Objective: <realistic long-form goal>"
    },
    {
      "role": "assistant",
      "content": "OBJECTIVE:\n...\n\nSUCCESS_CRITERIA:\n...\n\nCURRENT_STATE:\n...\n\nGLOBAL_STRATEGY:\n...\n\nTASK_QUEUE:\n[T1] ... (priority: high)\n[T2] ... (priority: medium)\n[T3] ... (blocked)\n\nLOOP 1\nSELECTED_TASK:\n...\n\nREASON_FOR_SELECTION:\n...\n\nEXECUTE:\n...\n\nRESULT:\n...\n\nCHECK:\n...\n\nLEARN:\n...\n\nUPDATE_STATE:\n...\n\nUPDATE_QUEUE:\n...\n\nDECISION:\nContinue / Retry / Replan / Stop\n\nLOOP 2\n...\n\nLOOP N\n...\n\nSTOP_REASON:\n...\n\nFINAL_OUTPUT:\n...\n\nPOSTMORTEM:\nWhat worked:\n...\nWhat failed:\n...\nWhat to improve next time:\n..."
    }
  ]
}
```

---

# 二、每个字段作用（非常关键）

---

## OBJECTIVE

总目标。

```text id="6tw7x7"
Launch AI newsletter in 7 days
```

训练模型理解 mission。

---

## SUCCESS_CRITERIA

什么叫完成。

```text id="g2rkfk"
Landing page live
First issue published
100 subscribers
```

训练模型目标导向，而不是瞎忙。

---

## CURRENT_STATE

当前进度。

```text id="w7d2lq"
0% complete
No branding yet
```

训练状态意识。

---

## GLOBAL_STRATEGY

总体打法。

```text id="2f17im"
Niche first, audience second, distribution third
```

训练高层规划。

---

## TASK_QUEUE

任务池。

```text id="jlwm9z"
[T1] Pick niche
[T2] Setup Beehiiv
[T3] Write issue #1
```

训练 backlog 思维。

---

## LOOP n

核心循环体。

每轮只做一件事。

---

## SELECTED_TASK

当前执行哪项。

训练优先级决策。

---

## REASON_FOR_SELECTION

为什么做它。

```text id="vv6hyu"
Highest leverage and blocks others
```

训练策略解释能力。

---

## EXECUTE

实际动作。

---

## RESULT

动作产出。

---

## CHECK

验证结果质量。

---

## LEARN

提炼经验。

```text id="v56z9n"
Specific niches outperform broad audiences
```

这是 evolution 核心。

---

## UPDATE_STATE

进度变化。

---

## UPDATE_QUEUE

任务队列变化。

```text id="e8k7ra"
[T4] Build referral incentive
```

训练动态任务生成。

---

## DECISION

四选一：

```text id="v2xyb7"
Continue
Retry
Replan
Stop
```

非常关键。

---

## STOP_REASON

为什么结束。

---

## FINAL_OUTPUT

最终交付物。

---

## POSTMORTEM

复盘能力：

* 什么有效
* 什么失败
* 下次如何更强

这是真 Evo 特征。

---

# 三、真实训练样例（简版）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Objective: Build a crypto market newsletter in 3 days."
    },
    {
      "role": "assistant",
      "content": "OBJECTIVE:\nBuild a crypto market newsletter in 3 days.\n\nSUCCESS_CRITERIA:\nLanding page live, first issue sent.\n\nCURRENT_STATE:\nNothing created.\n\nGLOBAL_STRATEGY:\nNiche fast-moving BTC traders.\n\nTASK_QUEUE:\n[T1] Pick brand\n[T2] Create landing page\n[T3] Draft issue #1\n\nLOOP 1\nSELECTED_TASK:\nT1\n\nREASON_FOR_SELECTION:\nBrand needed before landing page.\n\nEXECUTE:\nCreated name: DeltaEdge.\n\nRESULT:\nStrong trading identity.\n\nCHECK:\nMemorable and relevant.\n\nLEARN:\nShort names improve recall.\n\nUPDATE_STATE:\nBrand complete.\n\nUPDATE_QUEUE:\n[T2] Create landing page\n[T3] Draft issue #1\n\nDECISION:\nContinue\n\nSTOP_REASON:\nTime limit reached with MVP complete.\n\nFINAL_OUTPUT:\nBrand ready, landing page drafted, first issue prepared.\n\nPOSTMORTEM:\nWhat worked:\nFast niche selection.\nWhat failed:\nNo referral loop.\nWhat to improve next time:\nAutomate signup funnel."
    }
  ]
}
```

---

# 四、为什么这是最终版

因为它训练的不是回答能力，而是：

```text id="p6w0pb"
Operate
Adapt
Learn
Finish
```

---

# 五、你后续做 DPO 的 chosen / rejected 标准

## chosen：

* 会更新任务队列
* 会真正学习
* 会 replan
* 会停止得合理

## rejected：

* 假循环
* 固定模板
* 不学习
* 无限 continue

---

# 六、3000条数据建议分布

| 类型                        |  数量 |
| ------------------------- | --: |
| Coding agent tasks        | 900 |
| Research tasks            | 700 |
| Business execution        | 500 |
| Finance / trading ops     | 500 |
| Automation / productivity | 400 |

---

# 七、4090 单卡建议

因为 schema 长，推荐：

```text id="6f3hwd"
seq_len = 4096
batch = 1
grad_acc = 16
QLoRA
```

---

# 八、最关键一句话

> 普通数据训练模型回答问题。
> 这个 schema 训练模型推进现实世界目标。

---

# 九、如果你愿意，我下一步可以直接给你做：

## Claude 专用英文 Prompt（按这个最终版 schema 自动生成3000条）

## Unsloth 训练代码（适配 messages 格式）

## DPO 第二阶段数据格式

## Hugging Face 爆款发布页

只要你说：**继续**

