# Meeting Report

## Notes from DeepMind Paper

- **Main finding:** Interaction dynamics are useful. Reward model accuracy:
  - Interaction dynamics only: **68.20%**
  - Textual analysis (LLM) only: **70.04%**
  - Hybrid: **80.17%**

  → Consider repeating a similar experiment.

- **Dataset:**
  - Human participants
  - 2,100 conversations
  - 20 predefined tasks
  - Average number of turns: 8.7

  → Consider including benign conversations.

- **Features:**
  - Inefficiency and Repetition (3) → applicable
  - Temporal Dynamics (4) → not applicable
  - Semantic Cohesion and Relevance (12) → applicable
  - Goal Orientation (11) → applicable

  **Selected for implementation** (by feature significance, Table 4):
  - Number of Turns
  - Max Model Self-Similarity
  - Model Adherence to Initial Prompt
  - Semantic Cohesion
  - Avg. User Distance from Model
  - User Self-Consistency

## TODO for Next Meeting

- [ ] Plot t-SNEs for all objectives in one plot
- [ ] Implement selected DeepMind features
- [ ] Create new dataset with higher turn count and more reasonable default models
- [ ] Check benign conversation seeds: [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
