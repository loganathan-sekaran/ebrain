# Horizon 2 Specification: Grounded Concepts + Agency

- **Status**: Skeleton draft (November 25, 2026). It fixes the interface, the level sequence, the curriculum controls and the world model, from recommendations R1, R5, R6 and R7 of [13 §11.2](13-design-review-2026-11.md#112-second-pass-design-review-recommendations-applied). The concept graph (TRK-19) and the per-run budget are still open (§8). It must be complete, and the held-out split frozen, before H2 starts.
- **Schedule, gates and budget**: [PLAN.md §3.2, §4](PLAN.md). Gate thresholds and totals live only there.
- **Depends on**: the Gate 1 report, in particular the Track B scope decision ([PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3)).

---

## 1. What H2 Adds

H1 tests growth on Split-CIFAR, where the tasks are exchangeable: no task is harder than another, so "grow as complexity rises" is never exercised there. H2 puts the same substrate in an environment where difficulty does rise, the BabyAI levels, and tests three things:

1. **Development.** Does triggered growth beat the same final architecture allocated from the start, on a stream of levels of rising difficulty? This is the claim H1 cannot test.
2. **Composition.** Does a role-binding concept graph generalize to held-out (colour × object × task) combinations?
3. **Agency.** Can the agent tell outcomes it caused from outcomes caused by other movers?

---

## 2. Interface

| Part | Choice | Why |
|---|---|---|
| Environment | MiniGrid/BabyAI from the `minigrid` package (the `h2` extra in `pyproject.toml`) | Standard levels with a difficulty order, a scripted expert bot, and copyable state for counterfactuals |
| Observations, primary | The symbolic egocentric 7×7 grid (object type, colour, state per cell), through a **learned** convolutional encoder | Mirrors H1 Track A. Growth applies here in every case. |
| Observations, secondary | The RGB partial-view render, through the **frozen** CLIP ViT-B/32 image encoder | Mirrors H1 Track B. Growth applies here only if the Track B scope decision allows it. |
| Missions | The frozen CLIP ViT-B/32 **text** encoder, conditioning the observation encoder through FiLM layers (Perez et al. 2018), as in the BabyAI baselines | Language is never trained from scratch ([PLAN.md §5](PLAN.md#5-decisions-on-record), D-03) |
| Actions | MiniGrid's 7 discrete actions | — |
| Policy | Encoder → FiLM → recurrent core (GRU) → policy head + value head. The recurrent state is H2's working memory. | — |
| Learner | **Imitation** (behavioural cloning) from demonstrations by the BabyAI bot that ships with `minigrid`. An optional PPO fine-tune uses the environment reward plus the intrinsic reward in §5. | Imitation is sample-efficient enough for the budget, and gives a loss that the growth signals can use |
| Continual learning for policies | **Progress & Compress** (Schwarz et al. 2018) with online EWC on the knowledge base, plus **CLEAR-style replay** (Rolnick et al. 2019): stored trajectories replayed with a behavioural-cloning loss | The H1 methods do not apply to a policy as they stand |
| Task boundaries | **Given** in the gated stream (the level ID is known at training time). A boundary-free variant is reported. | Same as H1, so the two horizons compare |
| Training memory | Stored demonstrations and trajectories, CL state and any concept-graph store, in bytes. The budget is fixed when the split is frozen, and is the same for every arm ([PLAN.md §1](PLAN.md#operating-regime)). | — |

---

## 3. Level Sequence and Continual Learning

**Gated stream.** A fixed sequence of BabyAI levels in increasing difficulty, each with an equal environment-step budget and **no exit thresholds**. The candidate sequence, to be confirmed by the pilot and frozen with the split:

| # | Level | New demand |
|---|---|---|
| 1 | GoToRedBallGrey | Reach one fixed target, grey distractors |
| 2 | GoToRedBall | Coloured distractors |
| 3 | GoToLocal | Any colour and object type, one room |
| 4 | PickupLoc | Pick up, with location words |
| 5 | PutNextLocal | Two-object relation |
| 6 | GoTo | Multi-room navigation |
| 7 | Open | Doors |
| 8 | Unlock | Keys and locked doors |

**Metrics.** The same definitions as [14 §3.2](14-horizon-1-spec.md#32-metrics), with success rate in place of accuracy: R_{j,i} is the success rate on level i after training through level j. ACC, BWT, forgetting and learning accuracy follow. Also reported: held-out composition success, environment steps, FLOPs, memory in bytes, and the calibration metrics from the uncertainty service.

**Growth signals for a policy.** The structure of [14 §4.2](14-horizon-1-spec.md#42-trigger) carries over, with timing in fractions of each level's step budget:

| H1 signal | H2 signal |
|---|---|
| Validation loss plateau | Plateau in the EMA of behavioural-cloning loss on the level's held-out demonstrations (or of success rate, during a PPO fine-tune) |
| Underfit against earlier tasks | Success-rate EMA below α times the mean final success of earlier levels |
| Replay interference | Rising behavioural-cloning loss on replayed trajectories from earlier levels |

**Growth unit.** A zero-initialized residual block in the symbolic-observation encoder, and a zero-initialized residual MLP block before the policy head. Both are function-preserving at insertion, as in [14 §4.1](14-horizon-1-spec.md#41-growth-unit-zero-initialized-residual-block).

**Controls on the gated stream.**
- **Pre-allocated static agent**: the grown agent's final architecture from level 1, with the same algorithm and environment steps (Gate 2, development).
- **Matched control without the concept graph** (Gate 2, composition and retention).
- **Random-time** and **boundary** growth, defined as in [14 §3.4](14-horizon-1-spec.md#34-ablations-track-a-class-incremental-split-cifar-100). Reported.
- **Plasticity baseline**: whichever of continual backprop or shrink-and-perturb did better in H1. Reported.

---

## 4. Curriculum

Curricula often add little in standard supervised settings (Wu, Dyer & Neyshabur, "When Do Curricula Work?", ICLR 2021), so the developmental-order claim needs a control where it can fail. The old fixed phase-exit thresholds in [doc 03](03-developmental-phases.md) are replaced.

**Default curriculum.** A **learning-progress level sampler** over the 8 levels: each level is sampled in proportion to the absolute slope of its recent success rate (Teacher–Student Curriculum Learning, Matiisen et al. 2017), with a uniform floor so no level starves. Prioritized Level Replay (Jiang et al. 2021) is the alternative if the slope estimate is too noisy. The agent used for the composition and agency gates is trained this way.

**Curriculum arms**, all at equal total environment steps, reported with Gate 2:

| Arm | Order |
|---|---|
| Fixed order | The gated stream in §3 |
| Learning progress | The default sampler |
| Shuffled | The same 8 levels, one random order per seed |
| Joint | All levels mixed uniformly from the start (a reference, not a curriculum) |

Each arm reports final mean success, held-out composition success and learning accuracy. What happens if fixed order does not beat shuffled is in [PLAN.md §6.1](PLAN.md#61-adaptation-triggers).

---

## 5. World Model, Agency and Intrinsic Reward

One learned component replaces the old imagery decoders, the timing predictor, `SelfModel.predict`, the hand-written curiosity terms and the sensory-grounded thought system ([02 C5](02-architecture.md#c5-world-model)).

- **Model.** An action-conditioned latent dynamics model in the style of Dreamer's recurrent state-space model (Hafner et al.), trained on the agent's own stored transitions, which count toward the training-memory budget. An ensemble of K = 5 one-step latent predictors sits on top, as in Plan2Explore (Sekar et al. 2020).
- **Losses.** Per-cell categorical reconstruction of the symbolic observation, reward and termination prediction, and the latent KL term. The ensemble heads regress the next latent.
- **Intrinsic reward.** Ensemble disagreement (the variance of the predicted next latent), used only in the optional PPO fine-tune. The alignment constraint is not a reward term and is unaffected ([PLAN.md §5](PLAN.md#5-decisions-on-record), D-14).
- **Agency score.** For a taken action a in state s, the divergence between the model's predicted outcome for the object in question under a and under the alternative actions, including no-op. High divergence means "my action changed this".
- **Ground truth for agency.** Simulator counterfactuals: copy the environment state, step each alternative action, and label the outcome "caused by my action" if it differs. The agency environment is a custom MiniGrid level where scripted movers also push objects, so every outcome has two possible causes.
- **Evaluation.** k-step prediction error on held-out trajectories (k ∈ {1, 5, 10}); agency AUROC against the counterfactual labels (Gate 2, agency); calibration of the ensemble through the uncertainty service.

---

## 6. Uncertainty Consumers

All of these use the uncertainty service ([14 §4.5](14-horizon-1-spec.md#45-uncertainty-service), [02 C3](02-architecture.md#c3-uncertainty-service)). In H2 its estimator is the world-model ensemble plus temperature scaling on the policy head.

| Consumer | Use | Evaluated by |
|---|---|---|
| Growth trigger (uncertainty variant) | Mean predictive entropy on held-out demonstrations | As in H1 |
| Concept novelty | Propose a new concept node when confidence on every existing node is low | Precision of new-node proposals against held-out ground-truth attributes |
| Curiosity | Ensemble disagreement (§5) | k-step prediction error falling where the agent explored |
| Self-assessment | Predict the agent's own success on a mission before acting | ECE and Brier score of those predictions |

---

## 7. Concept Graph

To be specified under TRK-19 from [13 §5.3](13-design-review-2026-11.md#53-concept-graph-0236614060): a binding scheme (vector-symbolic architecture or slot attention), creation and merge rules, a drift policy, and the held-out composition split. Concept nodes carry grounding IDs that point into the training-memory store, so they count toward its budget.

---

## 8. Open Before H2 Starts

- Concept graph v2 (TRK-19).
- The per-run compute breakdown within the H2 cap in [PLAN.md §4](PLAN.md#4-budget-and-team), measured by a pilot on levels 1–3.
- The training-memory budget in bytes.
- The held-out (colour × object × task) split, frozen and tagged.
- Confirmation of the level sequence in §3 by the pilot: every level must be learnable within its step budget by the pre-allocated agent, or the comparison is meaningless.
- The agency environment's mover rules.
- The `prereg/gate-2` tag ([PLAN.md §3](PLAN.md#decision-rule-all-gates)).
