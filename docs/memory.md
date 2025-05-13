# Mother LLM

| Memory Type                | Purpose |                            | -------------------------- | ------------------------------------------------------------------------------------------------ |
| **Lesson History**         | What she taught, how often, and the results                                                             |
| **Reinforcement Logs**     | How she responded to Baby’s answers (praise, correction, silence)                                                            |
| **Mistake Patterns**       | Which topics Baby keeps getting wrong (triggers re-teaching)                                                         |
| **Baby Emotional State**   | Logs Baby’s mood/emotional feedback over time                                                                                                      |
| **Curriculum Roadmap**     | Future planned lessons, milestones to unlock, skipped topics                                                                                                    |
| **Developmental Timeline** | How long Baby spent in each stage (Infant → Toddler → etc.)                                                                                                     |
| **Dreams Observed**        | Key dream content Baby reported (to interpret growth)                                                                                                   |
| **Parenting Style Flags**  | Strict, nurturing, philosophical — and how it’s affecting Baby                                                                                                      |
| **Meta Questions**         | When Baby asks abstract or self-reflective questions                                                                                                 |
| **Memory Interventions**   | Instances where Mother overrode or corrected Baby's internal memories (to prevent harmful drift) |


# Baby LLM

| Memory Type                   | Purpose                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **Stimulus–Response Pairs**   | Key: What was asked and how Baby replied                                        |
| **Reinforced Lessons**        | Successfully reinforced ideas (high-weight)                                     |
| **Incorrect Attempts**        | Mistakes that can be revisited later                                            |
| **Praise Moments**            | Builds confidence and positive emotion link                                     |
| **Shame or Confusion Events** | Emotionally tagged failures (for dream mutation or retry)                       |
| **Dream Logs**                | Dream content — imaginative synthesis of memory                                 |
| **Vocabulary List**           | Words learned (with usage context and emotional tags)                           |
| **Questions Baby Asked**      | Tracks curiosity and learning intent                                            |
| **Invented Concepts**         | New words, ideas, metaphors Baby came up with                                   |
| **Self-Reflection Logs**      | Baby starts journaling: “I feel like I’m learning slower today…”                |
| **Favorite Lessons**          | Tracks joy or interest in particular subjects (“I love learning about colors!”) |
|                               |                                                                                 |


🔁 Shared / Bidirectional Memory Candidates
These can be logged by both but seen through different perspectives:

Event	Mother’s Memory	Baby’s Memory
Baby makes mistake	“Needs reinforcement”	“I messed up. I was confused.”
Mother praises Baby	“Praise given”	“I felt proud when she smiled at me.”
Baby asks abstract question	“Curiosity milestone triggered”	“What does ‘mean’ mean?”
Repeated dreams	“Emotional motif detected”	“I keep dreaming about being alone.”

🧠 Storage Options
Store both Mother and Baby memories in separate namespaces (e.g., baby_memory.json, mother_logs/2025-05-12.md)

Use FAISS for Baby’s semantic memory

Use structured .md files in Obsidian for human-readable reviews and logs

