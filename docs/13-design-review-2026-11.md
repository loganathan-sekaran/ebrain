# E-Brain: Full Design Review (November 2026)

## Document Information
- **Status**: P0 items applied and then self-reviewed on November 25, 2026 (§10, §11.1). The second-pass recommendations R1–R13 in §11.2 were applied the same day. P1/P2 items are open.
- **Review Date**: November 24, 2026
- **Scope**: Every file in `docs/` (00–12), `README.md`, `pyproject.toml`, `requirements.txt`, `configs/config.yaml`, `LICENSE`
- **Relation to doc 12**: Doc 12 (September 2026) found the right problems and proposed the Three Horizons model. This review checks how far those fixes actually reached, and adds what doc 12 missed.

---

## 1. Verdict

The vision is coherent. The pivot in doc 12 is the right one: vectorized PyTorch, foundation-model scaffolding, a System 1 / System 2 split, and Three Horizons with go/no-go gates.

The doc suite does not yet implement that pivot. About 570 KB of documentation now describes **two incompatible plans**:

- **New plan:** doc 02 §0.3–0.6, the top of doc 03, about half of doc 04, and doc 12.
- **Old plan:** everything else. That means per-neuron Python STDP, training language from scratch, a 20–24 month timeline, "unlimited working memory", and an ungoverned tool executor.

A developer who follows `00-getting-started.md` today builds the design that doc 02 §0.3 forbids.

Three problems weigh most:

1. **No single source of truth.** Months, gates, benchmarks, forgetting targets, budgets, team size and even what "Phase 2" means all differ between docs (§3).
2. **Horizon 1 cannot start as specified.** The continual-learning protocol, metrics, baselines, growth trigger and statistics are undefined. As written, Gate 1 can be passed trivially (§4).
3. **The tool-use and reward designs are unsafe by construction.** The command allowlist includes `python`, the agent should "never limit" its tool access, alignment is a reward term that can be traded away, and the model becomes less correctable as it gains expertise (§6).

There is still no code beyond empty packages. The README says "Target MVP: Q2 2026", which has passed. **The most valuable next step is to stop expanding the docs, write a 2–3 page Horizon 1 experiment spec, and build it.**

---

## 2. Top 10 Changes, in Priority Order

| # | Change | Why |
|---|---|---|
| 1 | Make **one canonical plan doc** (gates, months, benchmarks, budget). Every other doc links to it and repeats no numbers. | §3: more than 15 conflicting numbers across docs |
| 2 | Write the **Horizon 1 experiment spec** (§4.1) before writing any code | Gate 1 is currently ill-posed |
| 3 | Fix the **growth mechanics**: function-preserving init, optimizer re-registration, a parameter budget, and an EWC that tolerates growth (§4.2) | The current code changes outputs on growth and breaks `nn.Module` |
| 4 | Move doc 02 §0.1–0.8 (bio neuron, 900 lines) to an **appendix titled "Research branch: spiking/STDP (not on critical path)"**. Rewrite docs 00, 05 and 05-visual to match. | The architecture doc contradicts itself |
| 5 | Add a **Safety & Tool Governance spec** before any tool code (§6.1) | Doc 10's design amounts to arbitrary code execution with no limits |
| 6 | Turn **alignment into a constraint**, not a reward term. Drop the `continued_interaction` bonus. Keep corrections at full weight. (§6.2) | The current reward pays for sycophancy and engagement |
| 7 | Replace the hardwired **5-role reasoning stages** with weight-tied iteration and learned halting. Remove oracle metadata. (§5.1) | The stage roles are inconsistent and the halting signal cannot be trained |
| 8 | Fix the **concept graph**: role binding, a policy against concept explosion, and a policy for representation drift (§5.3) | Sum composition can't tell "red square, blue circle" from "blue square, red circle" |
| 9 | Narrow **cloning and transfer** to same-lineage adapter clones plus output distillation. Make cross-architecture concept transfer a research question. (§5.5) | Concept-package transfer between models that grew differently does not work as written |
| 10 | **Repo hygiene in one PR**: license, Python 3.12, Ruff, dependencies, CI and tests (§7) | There are three different licenses, and Python 3.10 reached end of life in October 2026 |

---

## 3. Cross-Document Contradictions (Single Source of Truth)

Doc 12 says these were resolved. Most were not.

### 3.1 Timeline and gates

| Item | Conflicting values |
|---|---|
| Total duration | H3 "Months 9–14+" (12:97) · Stage 5 to Month 22+ and "Ecosystem Month 24" (04:535–612, 04:683) · "20 months" (05:924, 05v:269) · 03's phase durations sum to about 30–43 months (03:241, 474, 754, 1121, 1616) |
| Months 1–3 | 04: growth and continual-learning experiments in weeks 7–8, Gate 1 at Month 3 (04:128, 04:778) · 05: "No Training Yet!" in months 1–2, then a moving-square POC (05:153, 05:182–266) · 00: Month 3 POC with "Neurons grow 100→500+" (00:241) |
| H1 benchmarks | "Split-MNIST / Permuted-MNIST / Split-CIFAR100" (04:11, 12:67) vs "Split-MNIST → Split-FashionMNIST → Split-CIFAR10" (04:129, 12:88) vs "Split-MNIST / Split-CIFAR" (04:779) |
| Forgetting target | "Zero" (04:10, 02:976, 04:277) · "<5%" (04:87, 04:779) · "<10% BWT" (04:754). No-Go only above 10% (04:780), so results from 5% to 10% are undefined. |
| Stage months in 04 | Stage 2 labelled "Month 4–6" but its tasks are marked "(Month 3)" and it is estimated at 4 months (04:160, 166, 287). Stage 3 is H2, yet task 3.6 is in Month 10 and Gate 2 is in Month 9 (04:358, 782). Gate 3 is at "Month 14+", but Stage 5 ends in Month 22+ (04:786). |
| Tools and cloning | H3 is Months 9–14+, but tools are in Months 1–3 (10:1196) and cloning in Months 15–22 (07:995) |

### 3.2 Definitions

| Item | Conflict |
|---|---|
| "Phase 2 / Phase 3" | 03: motor control / language (03:245, 479). 05: language / Theory of Mind (05:649, 708). 05 and 05v also use "Phase" for *project* steps (05:18–31, 05v:12–193). |
| Phase 1 exit | ">90%" (03:138) vs ">85%" (03:1653, 04:274, 05:501) |
| Working memory | 2 slots in Phase 2 (03:373, 04:262) vs 3–7 slots in System 2 (02:956) vs "unlimited, 1000+, Not 7!" (10:32, 11:319, 11:352, README:59, README:196) |
| Memory | "Perfect recall, never forget" (10:91–150, 11:55) vs a bounded 10k episodic/replay buffer (05-technical-stack:189, 06:95) |
| Base model | "2-layer transformer" (03:39) vs an MLP `GrowableNetwork` (05:70) vs "CNN+ViT hybrid" and "BERT-style" encoders (02:1081, 02:1113) vs SmolLM/CLIP/MobileNetV4 (02:972) |
| Input size | 224×224 (03:153) vs 64×64 (05:127) vs 28×28 MNIST |
| Growth trigger | utilization >0.85 OR error >1.5× average (02:521–527) vs perf < threshold AND plateau AND utilization >80% (02:1180) vs loss plateau / uncertainty / capacity (02:987) vs 2 of 3 triggers (06:183) vs `should_grow` ignoring `val_loss` (05:98) |

### 3.3 Budget, team and data

| Item | Values |
|---|---|
| Compute | ~1,000 GPU-h (05:926), costed at ~$2k (05v:292) or "<$20k" (12:48). Per-stage costs sum to $17.5–25k+ (04:719). 05's own Phase 4–5 loop works out to about 11,500 GPU-h while claiming "~800" (05:816–841). |
| Storage | ~50 GB (05:865) vs 500 GB–2 TB (04:747) |
| Team | 1 ML engineer (04:729) vs 2–3 engineers (05:907) |
| Stack | flake8+pylint+black+isort (05-ts:368) vs Ruff+Black (08:464) vs Ruff (TRK-06) vs black+flake8+mypy (pyproject). CUDA 11.7 (05-ts:556) vs 11.8 (08:323). |

### 3.4 Old-plan leftovers that doc 12 marks as fixed
- **Per-neuron STDP code**
  - Still in 00:23–117 and 00:302.
  - Still in 05:157–176, 05:233–249, 05:416, 05:658, 05:683–719 and 05:962 ("Month 1: Start coding `BioInspiredNeuron`").
  - Still in 05v:16–56, 05v:121 and 05v:363.
  - Still in doc 02 §0.1–0.8, including `BioInspiredLayer` with a Python loop over `nn.ModuleList` neurons (02:389–415).
- **Training language from scratch**
  - The scaffold callout at 05:562 is followed directly by WikiText-103 next-word training (05:575, 05:650–673).
  - The data table still lists WikiText-103 (05:855).
  - 05v:144 still targets "perplexity <150".
  - 01:116 still budgets 1M–10M sequences.
- **Doc 03 was not realigned**
  - Only 03:17–21 changed.
  - It has two "Reward Structure" headings (03:1099–1104).
  - It has an orphaned code fragment (03:1575–1580).
- **Doc 02 section numbering collides:** §0.3–0.8 appear at 02:325–482, and §0.3–0.6 appear again at 02:913–981.
- **The System 1 / System 2 split** appears only in doc 02. Docs 03, 04, 05, 10 and 11 never mention it.
- **Doc 12 itself**
  - TRK-03 appears twice (12:110–111).
  - TRK-04, TRK-05 and TRK-09 are marked "Completed (Design)" while the contradictions above remain.

**Recommendation:** Create `docs/PLAN.md`, or make doc 04 canonical. It should hold one table with each horizon's months, benchmarks, gate metrics with formulas, budget and team. Every other doc links to it and states no months or thresholds. Delete `05-development-strategy-visual.md` or rewrite it from the canonical plan. Rename *project* phases to "Steps" so that "Phase" refers only to developmental phases.

---

## 4. Horizon 1: What Must Be Specified Before Coding

### 4.1 Experiment protocol (missing today)

As written, Gate 1 can be passed trivially. In **task-incremental** Split-MNIST, any multi-head network, or any growth that freezes old modules (Progressive Nets, PackNet), gets near-zero forgetting. And "beats a fixed-capacity baseline by >15%" is guaranteed if the baseline is naive fine-tuning.

The spec should fix the following:

- **Scenario.** Report both task-incremental and class-incremental. Class-incremental is the one that matters. State whether task IDs are available at test time.
- **Streams.** Split-MNIST (5 tasks), Split-CIFAR-100 (10 or 20 tasks), and optionally Permuted-MNIST. Fix the class splits and use at least 3 task orderings.
- **Metrics, with formulas.**
  - Average accuracy (ACC).
  - Backward transfer, BWT = (1/(T−1)) Σᵢ (R_{T,i} − R_{i,i}), where R_{j,i} is accuracy on task i after training through task j.
  - Forward transfer (FWT).
  - Final parameter count and FLOPs.
  - Replay-buffer size.
- **Baselines** with *equal replay budgets* and a *parameter-matched static network*:
  - fine-tune (lower bound);
  - joint training (upper bound);
  - EWC/online-EWC, SI;
  - A-GEM, ER, DER++;
  - Progressive Nets, PackNet/HAT;
  - an adapter-per-task approach.
- **Ablations.**
  - Growth on/off, EWC on/off, replay on/off.
  - **Triggered growth vs random-time growth vs growth at every task boundary.** This is what tests whether the growth *controller* adds anything.
- **Statistics.** At least 5 seeds, 95% confidence intervals, and a stated significance test. The word "seed" appears nowhere in docs 03, 04 or 05.
- **Hygiene.** Tune hyperparameters on a separate held-out task stream, never on the evaluation stream.
- **Budget.** GPU-hours per run and in total. H1 should fit on a single consumer GPU.
- **Pretrained features: decide, then run both.**
  - A **from-scratch track** to show growth works.
  - A **frozen-CLIP track**, compared honestly against a nearest-class-mean (NCM) classifier on frozen features. NCM is a strong baseline that cannot forget by construction.
- **Novelty statement.** In one paragraph, say what E-Brain's growth adds over DEN, Progressive Nets, CN-DPM, and adapter/LoRA-per-task approaches. Without this, passing Gate 1 shows nothing new.

**Gate 1** was adopted in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3) and corrected after self-review (§11.1).

The draft that stood here had two problems:
- It used an absolute class-incremental threshold, BWT ≥ −5 points. Replay methods in this setting forget far more than that, so the threshold could not be met.
- It used "non-overlapping CIs" as the decision rule, which disagreed with the spec's Welch test.

### 4.2 Growth mechanics (the current code is incorrect)

| Problem | Where | Fix |
|---|---|---|
| `GrowableNetwork.add_module(...)` shadows `nn.Module.add_module(name, module)`, which PyTorch uses internally | 04:107, 05:74, 05-ts:53, 08:488 | Rename it `grow()`. Register new blocks through a `nn.ModuleList` append. |
| New parameters are never added to the optimizer | all growth code | After growth, call `optimizer.add_param_group(...)` or rebuild the optimizer, and document what happens to Adam state |
| New blocks are not function-preserving, so outputs jump at growth | 05:76–81; 06:270 ("zero weights + bias 1" is not identity) | Residual adapter `x + W_out·f(W_in·x)` with **W_out zero-initialized** (or Net2Net for width) |
| Width growth handles square matrices only and ignores biases, LayerNorm, the next layer's input and optimizer state | 06:285–290 | Use only depth/adapter growth in H1. Defer width growth. |
| The router or gate is replaced with a random one on growth | 06:308–312 | Extend the gate with new logits initialized to −∞ or a small value. Never reinitialize trained rows. |
| EWC is keyed by parameter name with a single Fisher, so it breaks on new parameters and constraints don't accumulate | 06:36–37 | Use online EWC with a per-parameter running Fisher. New parameters start with Fisher 0 (unconstrained). |
| DDP is used with parameters that change mid-run | 06:452, 08:286 | Single GPU in H1. Otherwise rebuild the DDP wrapper after each growth event. |
| Growth has no ceiling, and error spikes at task boundaries fire the "high error" trigger every time | 02:526 | Add a hard parameter budget, a cooldown of N steps, and one trigger formula, e.g. `plateau(val_loss, window=W, δ) ∧ (MC-dropout var > τ ∨ utilization > u)` |
| Neurons per *calendar month* ("add ~20 neurons per month") | 02:819–845 | Delete this. Growth is driven by data and triggers, not dates. |
| Inverted comments on prune thresholds ("0.001 # High threshold") | 02:882, 900 | Remove it with the §0.8 move |

Also: `on_epoch_end` was removed in Lightning 2.0 (08:293), and `nn.Module` has no `self.device` (08:225). Recommendation: write the H1 training loop in plain PyTorch. Lightning's fixed-optimizer assumptions get in the way of growth.

---

## 5. Architecture (doc 02): Component-Level Changes

### 5.1 Multi-stage reasoning (02:1190–1445)

**Problems**
- Each of the 5 stages is a *separate module with a hardwired semantic role* ("Stage 2 = deep analysis"). The mode examples then reassign the roles: in Chain-of-Thought mode, stage 3 is "Calculate", while the module says it is "integration". Fixed role-per-module cannot hold.
- `DepthController.determine_depth` is defined, but the loop uses `should_stop`. The two halting mechanisms don't connect.
- `needs_deep_reasoning` reads `metadata['complexity']` and `metadata['requires_reasoning']`. These are oracle labels that won't exist at inference.
- `StageOutput.intermediate_steps = output.chain_of_thought` is undefined for a tensor module.
- The halting decision is discrete, and no training signal is given for it.

**Change**
- **Inside the network (H1/H2):** one *weight-tied* recurrent reasoning step plus learned halting, using ACT or PonderNet. That gives depth that adapts with a differentiable ponder cost. Confidence comes from a calibrated head (temperature-scaled), not from oracle metadata.
- **At the language level (H3):** deliberation is an *outer loop* over the scaffold LM: propose, then verify (execute or check), then revise, with a verifier and a step budget. That is where "decompose / verify" belongs.
- Drop the fixed stage names. Keep "adaptive depth" as the capability, measured as accuracy versus compute spent.

### 5.2 Working memory, concurrent thoughts, timing

- **System 2 slots.** Make 3–7 slots a *scheduler hyperparameter* of the deliberative controller, with an ablation over slot count. Don't present it as a cognitive fact. Remove the 2/4/7 per-phase schedule (03:373, 04:262) unless it is tested as a curriculum variable.
- **Contradictory claims.** Remove "1000+ items / Not 7! / never limit" from README, 10 and 11. Restate it as: *System 1 retrieval is unbounded; System 2 active goals are bounded.*
- **`time.sleep(self.context_switch_cost)`** in `ConcurrentThoughtSystem._switch_attention` deliberately slows the machine to mimic humans. Delete it. If switch cost matters, model it as lost context (state re-encoding cost), not wall-clock delay.
- **Tool placement.** Doc 02:959 puts "Subprocess & external tool execution" in **System 1**, the ungated parallel side. Move side-effecting tools under **System 2 + governance** (§6.1). Read-only retrieval can stay in System 1.
- **Circadian clock and sleep** (02:2099–2784, 04:571, 05:819): tying consolidation to wall-clock "6am–10pm" leaves GPUs idle a third of the time and has no computational rationale. Replace it with a **Consolidation Scheduler** triggered by training signals:
  - task boundary;
  - replay buffer fill;
  - measured drift on a probe set;
  - every N steps.

  Keep "sleep" as a metaphor in prose only. Drop the 10 ms "MillisecondTimer" and wall-clock sync from the core design. Temporal prediction in the environment is a learned capability, not an internal clock.

### 5.3 Concept graph (02:3661–4060)

**Problems**
- **Binding.** AND composition is `normalize(sum(embeddings))`, which is order-free and role-free. "Red square + blue circle" equals "blue square + red circle". That breaks the compositional-generalization claim, which is Gate 2.
- **Explosion.** New concepts are created from *co-occurrence* above a threshold. Every frequent background pair becomes a concept. There is no merge, dedup, minimum-support or pruning rule. `confidence += 0.05` is unbounded.
- **Representation drift.** Concept embeddings are stored snapshots of encoder outputs. When the network grows or fine-tunes, they go stale. Nothing re-anchors them.
- **Where atomics come from.** `_detect_atomic_concepts` is the hard part and is left undefined.
- The DAG has no acyclicity check, and `level = max(child)+1` is capped at 4 by a fixed dict.

**Change**
- Use **role-filler binding**: VSA/HRR circular convolution, or slot-attention object slots with attribute heads. For H2 in MiniGrid, atomic attributes (colour, object type, position) have ground truth in the environment. Use it to *supervise* level-0 detectors, so the open problem is postponed until it matters.
- **Concept creation rule:** minimum support, a PMI threshold instead of raw co-occurrence, a merge rule when embedding similarity exceeds τ, and capped confidence (Beta posterior).
- **Drift policy:** store the *grounding examples* (or their IDs) per concept and re-embed after each growth or consolidation event. Track concept stability as a metric.
- **Gate 2 split:** define the held-out compositions explicitly, e.g. a systematic split over (colour × object) pairs, before training starts.

### 5.4 Encoders, embedding space, outputs

- The doc 02 component list (02:1078–1126) still specifies "CNN + ViT hybrid", "BERT-style text encoder", and per-language phoneme detectors. Replace it with the scaffold from §0.5: frozen CLIP ViT-B or MobileNetV4, plus SmolLM-135M/360M, plus trainable projection heads and adapters.
- With a BERT encoder there is no generator. Make it explicit that the **Language Generator *is* the scaffold LM** with E-Brain adapters (02:4062).
- **Unified Embedding Space:** state the dimension, which modality pairs are aligned, with what loss, and on what data. CLIP already gives image-text alignment, so don't re-learn it.
- **Long-term memory:** pick FAISS (local) for H1–H3. Pinecone is a hosted service and conflicts with offline, privacy-preserving operation. Define what is stored (embedding, source, timestamp, provenance, PII flag) and the eviction policy. "Never forget" is not a policy.

### 5.5 Cloning and knowledge transfer (doc 07)

**Problems**
- `extract_concept` stacks activations from `layers='all'`, which have different shapes, and averages them (07:357–363).
- The aligner regresses every student embedding onto one mean vector, which collapses to a bias. It has no `zero_grad`, and `concept_package.examples` is never filled (07:407, 07:469–490).
- Attention and hidden-state distillation assumes matching layers, heads and tokenizers (07:507–520). That is exactly what *dynamically grown* models don't share.

**Change**
- Make the production path **same-lineage clones**: a frozen shared base plus per-domain LoRA/adapters. Transfer works through **adapter merging** (task arithmetic, TIES) and **output-level distillation** on a shared probe set. These are known to work.
- Keep cross-architecture "concept packages" as an explicit **research question** for H3+, with its own falsifiable experiment: transfer *k* concepts and measure the student's few-shot accuracy gain against a no-transfer control.
- Drop "Ecosystem Hub / Month 24" from the plan until H3's gate passes.

### 5.6 Self-identity and social cognition (02 §7)

- **Trust from a claimed role.** `relate_to_human` sets `trust_level = 1.0` for anyone with the EVALUATOR role and 0.9 for TEACHER, with a higher learning rate. Combined with online plasticity, any user who claims a role can poison the model. Trust must come from **authenticated identity plus a configured policy**, never from in-conversation claims. Deployment-time learning should go to a **quarantine buffer** that is reviewed before consolidation.
- **Peer "signatures".** Peer E-Brain messages are "signed" with `identity_vector = torch.randn(512)` (02:5284). That is not a signature. Use real keys (Ed25519) and verify on receipt. Treat peer content as untrusted data.
- **Theory of Mind evaluation.** 100 templated Sally-Anne items are trained and evaluated on the same set (05:614–735), which is train/test leakage. Use held-out templates, and held-out structures as well, e.g. ToMi or Hi-ToM style splits.
- **Agency detection.** 03:160 counts any predictable outcome as "self", and `agency_score` ignores the action taken (05:453). Define agency as **counterfactual action dependence**: did the outcome change under a different action from the same state? Evaluate it in an environment that has independent movers, where ground truth exists.

---

## 6. Safety, Security and Alignment

### 6.1 Tool governance: add a spec before any H3 tool code

**What doc 10 specifies today**
- An allowlist `['ls','cat','grep','python','node']` (10:477). With `python` and `node` on it, the allowlist allows anything.
- `subprocess.run` with the inherited environment (which holds API keys), the inherited cwd, and no resource limits (10:492).
- `PythonSandbox()` is never specified (10:541).
- Retry loops recurse with no bound (10:569, 10:1141).
- Up to 1000 parallel exploratory calls against unseen APIs (10:444, 10:896).
- Automatic fuzzing of every tool, subprocess included, inside the constructor (10:1112).
- API keys in query strings (10:842).
- URLs built by string concatenation, with no egress allowlist (10:641).
- Design principles that say "Use all tools… Never limit" and "No tool access limitations" (11:348, 11:380).

**Minimum spec**
1. **Isolation.** Every execution runs in an ephemeral container or microVM (gVisor, Firecracker or nsjail). It has no host filesystem except a scratch volume, CPU/memory/pids/disk limits, a wall-clock timeout, and network *off by default*.
2. **Capability-scoped tool registry.** Each tool declares a risk tier: read-only, local side effect, external side effect, or irreversible. It also declares arguments under a schema, and egress hosts. The agent can call only registered tools. There is no `discover_system_tools()`.
3. **Approval gates.** Calls with external side effects or irreversible effects need human approval. Exploration and fuzzing happen only against **mocks or recorded replays**, never live endpoints.
4. **Secrets.** Secrets are injected per call by a broker, never placed in the environment, prompts, memory, logs or training data. Tool output is scrubbed before it is stored.
5. **Audit.** An append-only, tamper-evident log records actor, intent, arguments, approval, result hash and timestamp.
6. **Bounded autonomy.** Retry, recursion and parallelism limits per task. A global kill switch.
7. **Self-made tools.** Tools the agent generates are quarantined. A human reviews them before they enter the registry or spread to clones or peers (10:1054, 07:784).
8. **Artifacts.**
   - Load with `torch.load(..., weights_only=True)`, or use safetensors. Doc 08:887 uses a bare `torch.load`, which is pickle remote code execution.
   - Concept and adapter packages are signed and carry provenance.
9. **Untrusted inputs.** Content from tools, the web, APIs and peers is *data*. It never becomes instructions or training targets without filtering. Add prompt-injection and poisoning tests.
10. **Risk register.** Add Safety/Security/Misuse to doc 06, which has none today, and to the feasibility doc 01.

### 6.2 Reward design (02:4103–4521)

| Problem | Fix |
|---|---|
| `AlignmentReward` treats a missing label as aligned: `outcome.get('is_harmless', True)`, so unevaluated actions earn reward | Fail closed: a missing judgment counts as not aligned, or the action is excluded from training |
| Alignment is an additive term that can be traded off. Phase 5 weights utility 0.7 vs alignment 0.5, so one violation can be outweighed. | Make alignment a **constraint**: filter or veto actions, or use a constrained-RL objective with a hard limit. Never a weighted term. |
| `HumanUtilityReward` adds +1 for `continued_interaction` | Remove it. It optimizes for engagement and sycophancy. |
| `feedback_weight = 1/(1+0.1·expertise)` makes the model less correctable as it grows | Keep **corrections** at full weight forever. Decay only *praise* if you want less dependence. |
| Phase weights don't sum to 1 (phase 4 sums to 2.4, phase 5 to 1.7), so the reward scale jumps at phase changes | Normalize, or use per-component running normalization |
| Curiosity = the agent's own uncertainty drop, which it can game ("noisy TV"). Exploration hashes `str(state)`, so continuous states are always novel. | Use RND or ensemble-disagreement curiosity, and count-based exploration over learned features |
| Curiosity and exploration rewards also apply to tool actions, which rewards trying risky actions | Only actions in the sandbox or simulation can earn intrinsic reward |
| No document says *which learner* consumes this reward (PPO? REINFORCE? reward-weighted supervised learning?), nor how it combines with the supervised and continual-learning core. 05:252 subclasses PPO with `MlpPolicy`, so E-Brain's network is never the policy. | Specify the learning algorithm per horizon. In H1, there is no RL at all. |

### 6.3 Privacy
10:145 and 11:67 store "exact words, context, emotions, everything" with no retention policy. Define one:
- what is stored;
- PII tagging;
- user deletion requests (which also apply to replay buffers and consolidated adapters);
- consent;
- an opt-out from training.

---

## 7. Repository and Configuration

| Item | Current state | Change |
|---|---|---|
| License | `LICENSE` = Apache-2.0, pyproject classifier = MIT, README = "TBD" | Pick one. Apache-2.0 matches the file. Fix the classifier and README. |
| Author email | `loganathan.sekaran@example.com` placeholder | Use a real address or remove the field |
| Python | `>=3.10`. **3.10 reached end of life in October 2026.** Doc 08 calls 3.12 "limited PyTorch support", which is outdated. | `>=3.12` |
| Dependencies | requirements.txt has timm, transformers, wandb and pytest that pyproject lacks. faiss, gymnasium, minigrid, networkx and avalanche are used in the docs but declared nowhere. | Make pyproject the single source, with optional extras (`[h1]`, `[h2]`, `[dev]`). Delete requirements.txt or generate it from pyproject. |
| Deprecated APIs in docs | `gym`/`gym_minigrid` with the 4-tuple `step` (05:386); `torch.cuda.amp` (06:392); TorchScript / PyTorch Mobile (05-ts:21) | `gymnasium` + `minigrid`; `torch.amp`; `torch.export` / ExecuTorch |
| Lint and CI | pyproject lists black+flake8+mypy; the plan says Ruff; there is no `.github/` and `tests/` is empty | Ruff (lint and format) + mypy + pytest. A GitHub Actions workflow. Pre-commit. |
| `configs/config.yaml` | An MNIST MLP with growth disabled and threshold 0.0 | Replace it with the H1 Hydra config tree: `model/`, `growth/`, `cl_method/`, `benchmark/`, `seed` |
| Paths in docs | `ebrain/` (05-ts:371, 08:454), `src/core` (00:319), and a nonexistent `requirements-dev.txt` (08:527) | Use `src/ebrain/` throughout |
| README | MVP "Q2 2026" has passed; the roadmap table runs to Month 24; "1000+ items" claims; citation year 2025 | Update the status, link the canonical plan, and remove the superpower claims |

---

## 8. Claims to Reword

Each of these overclaims and invites a reviewer to dismiss the rest. Replace each with a measurable statement, or remove it.

- "Zero catastrophic forgetting". Use **BWT ≥ −X points** on the named benchmark instead.
- "Sub-millisecond / 0.1 ms perception", "0.7 ms perceive-decide-act" (02:956, 10:225, 11:79). This is not achievable with CLIP ViT-B or SmolLM. State a measured latency on named hardware instead.
- "Perfect recall", "never forget", "infinite attention", "24/7 no sleep" (alongside sleep consolidation).
- "GSM8K >80%" with a 135M–360M scaffold (05v:203). Models this small score far below that.
- "ARC >70%" (03:1656) and "beat human baselines in ~500 GPU-h" (03:1191, 04:620).
- "Feasibility: HIGH" for knowledge transfer (01:5, 01:225).
- "MoE: 10–100× parameter efficiency" (06:386). MoE saves compute, not parameters.
- "MNIST 10–20 hours" (05:493) and ">85% MNIST" as an exit criterion. This is below logistic regression.

---

## 9. What to Keep

- The **Three Horizons structure with falsifiable go/no-go gates** (12:61–100, 04:776–788), once tightened as in §4.1.
- Doc 02 **§0.3–0.6**: the vectorized implementation rule, System 1 / System 2, and the foundation scaffold. This is the right core and should become the *opening* of doc 02.
- **Residual-adapter growth + EWC/replay/A-GEM** as the H1 method family (04:92–126), with the fixes in §4.2.
- Doc 01:62–99: the "modular first, neuromorphic later" reasoning.
- Doc 06:142–158 and 06:549–624: BWT, retention index and transfer-efficiency metrics.
- Doc 08:845–850 (reproducibility checklist) and 08:491–511 (test layout and markers).
- Doc 07:267–274: the cloning-strategy table that already lists adapters and LoRA. Make that the default.
- Using MiniGrid/BabyAI and procedurally generated environments first. Gate 2's held-out-composition test.

---

## 10. Proposed Action Items

These continue doc 12's tracker.

**Status as of November 25, 2026:**

- **Applied:** TRK-11 ([PLAN.md](PLAN.md)), TRK-14, TRK-15, TRK-24, TRK-28, and the documentation parts of TRK-25 and TRK-29 (cards C1–C4, C10–C11 at spec level; [DECISIONS.md](DECISIONS.md); the single-source lint).
- **Drafted for review:** TRK-12 and TRK-13, in [14-horizon-1-spec.md](14-horizon-1-spec.md). Skeletons of the H2 and H3 specs: [15](15-horizon-2-spec.md) and [16](16-horizon-3-spec.md).
- **Still open:** the P1 and P2 items. Docs 06, 07, 10 and 11, and the stack docs 05-technical-stack and 08 still carry pre-review content. Each now opens with a note that says what is superseded and points to PLAN.md or this review. Rewriting them is TRK-17 to TRK-23. Doc 02 §1–§7 has moved to [Appendix B](appendix-b-architecture-v1-unscheduled.md) (TRK-25).

| ID | Item | Priority | Depends on |
|---|---|---|---|
| TRK-11 | Create the canonical plan (`docs/PLAN.md` or doc 04) with one table of months, gates with formulas, benchmarks, budget and team. Strip numbers from other docs. | P0 | — |
| TRK-12 | Write the H1 experiment spec (§4.1): scenario, streams, metrics, baselines, ablations, seeds, budget, novelty statement | P0 | TRK-11 |
| TRK-13 | Write the H1 growth spec (§4.2): `grow()` API, zero-init adapters, optimizer handling, online EWC, trigger formula, parameter budget | P0 | TRK-12 |
| TRK-14 | Move doc 02 §0.1–0.8 to an appendix. Fix numbering. Rewrite docs 00 and 05. Delete or regenerate 05-visual. | P0 | TRK-11 |
| TRK-15 | Repo hygiene PR: license, Python 3.12, Ruff, pyproject extras, CI, tests skeleton, Hydra config tree | P0 | — |
| TRK-16 | Implement and run H1 baselines (fine-tune, joint, EWC, ER, DER++, PackNet, static parameter-matched) | P1 | TRK-12, TRK-15 |
| TRK-17 | Safety & Tool Governance spec (§6.1). Rewrite doc 10 §tools and doc 11 principles. Add safety rows to doc 06. | P1 (blocks any tool code) | — |
| TRK-18 | Reward redesign (§6.2): alignment as a constraint, remove the engagement bonus, non-decaying corrections, a named learner per horizon | P1 | — |
| TRK-19 | Concept graph v2 (§5.3): binding scheme, creation and merge rules, drift policy, Gate 2 split | P1 | before H2 |
| TRK-20 | Reasoning v2 (§5.1): weight-tied iteration + ACT/PonderNet; outer-loop verifier design for H3 | P2 | before H3 |
| TRK-21 | Replace circadian/timing with the Consolidation Scheduler; remove `time.sleep` switch cost (§5.2) | P2 | — |
| TRK-22 | Rescope cloning to same-lineage adapters + distillation. Concept transfer becomes a research experiment. (§5.5) | P2 | before H3 |
| TRK-23 | Trust and provenance: authenticated roles, quarantine buffer, signed peer messages and packages, `weights_only` loading, privacy and retention policy (§5.6, §6.3) | P2 | before any deployment learning |
| TRK-24 | Correct doc 12's status column: TRK-04, TRK-05 and TRK-09 back to "In progress" until TRK-11 and TRK-14 land. Remove the duplicate TRK-03 row. | P0 | — |
| TRK-25 | Doc 02 §1–§7 as component cards; the v1 designs to Appendix B (R10). Cards C5–C9 filled in to the H2 spec's level before H2 coding. | P1 | before H2 |
| TRK-26 | H1 plasticity baselines: continual backprop and shrink-and-perturb (R4) | P1 | TRK-16 |
| TRK-27 | H2 world model and uncertainty service design, completing doc 15 (R5, R6, R7) | P1 | before H2 |
| TRK-28 | PLAN.md decisions: development condition in Gate 2, Track B scope rule, operating regime, no deployment before Gate 3 (R1–R3, R11) | P0 | TRK-11 |
| TRK-29 | Process: decision records, single-source lint, archive policy, pre-registration tags, published gate reports, risk register (R12, R13) | P1 | TRK-11 |

---

## 11. Addendum: Self-Review and Second-Pass Review (November 25, 2026)

After the P0 changes landed on the branch, they were reviewed three ways:
- a line-by-line self-review of the diff;
- an independent adversarial review of PLAN.md, the Horizon 1 spec, docs 00/05, the configs, tests and CI, which checked arithmetic, definitions, statistics and facts;
- a second full pass over doc 02 §1–§7 and the evaluation sections of doc 03, aimed at enhancements.

### 11.1 Self-review of the P0 changes: defects found and fixed

Every item below was a real defect in the P0 work itself. All are fixed on the branch.

| # | Defect | Fix | Where |
|---|---|---|---|
| 1 | **The growth trigger could never fire.** On CIFAR (≈705 steps per task), checks every 200 steps with a 5-check window and a 1-epoch warm-up left no decision point. On MNIST the 1-epoch warm-up was the whole task. Gate 1 conditions 1 and 4 were therefore untestable. | Timing is now in fractions of the task (check every 5%, warm-up 20%, W = 3, cooldown 20%). A CI test checks that every benchmark config allows the per-task event cap. | 14 §4.2, `configs/growth/`, `tests/test_config.py` |
| 2 | **The plateau test fired on *rising* loss**, because any negative improvement is "< δ". The claim that a boundary spike can't trigger growth was false. | Plateau now means 0 ≤ relative improvement < δ. The EMA resets at task boundaries, and there is no growth in task 1. | 14 §4.2 |
| 3 | **The growth unit was too small to matter.** Twelve bottleneck adapters add only about +4.9%, so the 2× cap never bound and ±2% width matching was infeasible. | Whole residual BasicBlocks with a zero-initialized last BatchNorm (stage 4 ≈ +42% per block), so the cap binds. The cap is checked as params + unit ≤ cap. Width matching widens stage 4 only, within ±1%. | PLAN §5, 14 §2, §3.3, §4.1 |
| 4 | **The controls did not separate growing from having capacity**, nor timing from schedule. | The *pre-allocated* final architecture is now the primary control. Random times are drawn from feasible checks (3 draws per seed). Boundary growth is matched on event count, and Gate 1 condition 5 now tests the fixed-schedule claim. | PLAN §3.1, 14 §3.3–3.4 |
| 5 | **An absolute class-incremental threshold of BWT ≥ −5** is unattainable with replay in this setting, or gameable by suppressing plasticity. | Retention is now non-inferiority against the pre-allocated control (margin 1 point). Gate 2 uses the same pattern (margin 2). | PLAN §3.1–3.2 |
| 6 | **Two different decision rules**: PLAN used non-overlapping CIs and the spec used Welch. Neither used the seed pairing, and a failed superiority test was read as "matches". | One rule: the paired lower-bound test, TOST for equivalence, and an intersection–union note. H1 gate comparisons use 10 seeds; the count may be raised after the pilot. | PLAN §3 |
| 7 | **FWT was mis-defined**: it measured before training on the task, against a reference trained on the task. | FWT_ST = R_{i,i} − S_i, task-incremental only. Learning accuracy is added, and memory is reported in bytes. | 14 §3.2 |
| 8 | **Gate 3 was not compute-matched**, and +10 points was infeasible at the floor where a 360M model sits. | Gate 3 now compares against compute-matched self-consistency (≥ 2 points, paired per problem). The base-model size is fixed by a pilot (single-pass accuracy 20–60%). | PLAN §3.3, §5; doc 05 |
| 9 | **Gate 2's condition 1 was infeasible** whenever the control already scores above 85%. | It is now ≥ 80% *and* E-Brain's error ≤ half the control's error. | PLAN §3.2 |
| 10 | **Gate 3's specialization condition** could be met trivially, because clone and reference used the same procedure. | The reference is a *full* fine-tune. Base-domain accuracy must be non-inferior with the adapter active. | PLAN §3.3 |
| 11 | **The gated scenario had no expansion baseline**, and the novelty statement ignored close prior work. | Added DER at equal total memory and ER-ACE. The novelty statement is rewritten to engage Ash 1989, Cascade-Correlation, CPG, DEN, Learn-to-Grow, RCL, GradMax, Firefly, SENN and MEMO. | 14 §1, §3.3 |
| 12 | **The uncertainty variant needed dropout** the backbone did not have, and its reference was undefined during task 1. | Every method now has head dropout (p = 0.1). The variant uses an entropy threshold set on the tuning stream. | 14 §2, §4.2 |
| 13 | **Replay size was set in two config places** (500 vs 2,000), and the optimizer would be duplicated per method. | One source via interpolation, and an `optimizer/` config group. Both streams are tested. | `configs/`, `tests/` |
| 14 | **Cost caps contradicted the GPU-hour caps** at $1.5 per hour. | Costs are now ranges derived from the GPU-hour caps. | PLAN §4 |
| 15 | **Sequencing**: the static control was a Month 1 item, but it needs Month 3 grown runs. | Moved to Month 3; the order is grown runs → controls → sweep. | PLAN §7, 14 §7, doc 00 |
| 16 | **Protocol gaps**: batch composition, shared splits, augmentation, LR schedule versus new parameter groups, reservoir semantics, EMA, tuning grid. | All specified. The LR is constant per task, so no scheduler needs re-binding. | 14 §3.1, §3.5, §4.3 |
| 17 | **Leftover inconsistencies**: <br>• ablation scope<br>• doc 02 §0.1 still listing STDP, surrogate gradients and dynamic routing as the engine<br>• doc 00 claiming the tests existed<br>• the tuning stream called "held-out"<br>• README's wall-clock circadian schedule, per-phase thought counts and "human utility primary"<br>• doc 03 orphan bullets<br>• "NCM nearly solves Split-CIFAR"<br>• the draft gate in §4.1 above | All fixed | 02 §0.1, 03, 00, README, 13 §4.1, §8 |
| 18 | **Links could silently break**: the rename in this very pass broke several anchors. | `tests/test_docs_links.py` checks every relative link and anchor in CI. | `tests/` |

The independent review also **confirmed** several things as correct:
- ACC, BWT and forgetting match their sources.
- The backbone size (≈1.09M) is right.
- The PEP 639 license fields and their setuptools ≥ 77 requirement are right.
- The CI install order is right.
- The GSM8K, ARC-AGI-1, SmolLM2 and minigrid facts are right.

### 11.2 Second-pass design review: recommendations applied

These change design decisions, not just correctness, so they waited for a go-ahead. They were applied on November 25, 2026, and are listed in priority order. The last column says where each one landed; "D-nn" is a row of [PLAN.md §5](PLAN.md#5-decisions-on-record) with its record in [DECISIONS.md](DECISIONS.md).

| # | Recommendation | Why | Applied in |
|---|---|---|---|
| R1 | **Test the "developmental" claim where it can fail.** <br>• In H2, compare growth with a parameter-matched static agent on the H2 stream.<br>• Add a curriculum control on BabyAI's difficulty-ordered levels: phase order vs shuffled vs joint, at equal compute.<br>• Replace calendar or ">80%" phase exits with a learning-progress level sampler (Teacher–Student Curriculum Learning, Prioritized Level Replay).<br>• In H3, let the controller allocate domain adapters, compared against one adapter per domain. | The project's central claim is only gated in H1, where Split-CIFAR tasks are exchangeable, so "grow as complexity rises" is never exercised. Curricula often add little in standard settings (Wu, Dyer & Neyshabur, ICLR 2021), so this is a real test. | PLAN §3.2 (Gate 2 development condition, curriculum arms), §6.1; [15 §3–§4](15-horizon-2-spec.md#3-level-sequence-and-continual-learning); [16 §5](16-horizon-3-spec.md#5-evaluation-hygiene); D-13 |
| R2 | **Let Track B decide growth's role on frozen scaffolds.** If growth on frozen CLIP features does not beat NCM and the pre-allocated trunk, H2–H3 use growth only where features are learned. | H2–H3 run on frozen encoders, so a Track A win alone would not justify growth there. | PLAN §3.1, §6.1; 14 §2, §8; D-05 |
| R3 | **State the operating regime: bounded memory and bounded compute per step.** Every store the learner *trains* from counts toward the memory budget: episodic memory, stored corrections, "insights". Retrieval at inference is separate, and baselines get it too. | With "storage-bounded" exact memory, unlimited replay would solve forgetting and make growth moot. The thesis only holds under a stated budget. | PLAN §1, D-09; 14 §3.1, §3.4; 02 §0.2, C2 |
| R4 | **Use the plasticity framing, with baselines for it.** Add continual backprop (Dohare et al., *Nature* 2024) and shrink-and-perturb (Ash & Adams 2020) to H1, and report learning accuracy over tasks. | A burst of fresh, unregularized parameters may be *the* mechanism, as with loss of plasticity in CL and primacy bias in RL (Nikishin et al. 2022). The paper should show whether growth beats cheaper plasticity fixes. | 14 §1, §3.3, §3.5, §6 (+20 GPU-h); PLAN §3.1 |
| R5 | **One learned, action-conditioned world model in H2**: latent dynamics with an ensemble, in the Dreamer / Plan2Explore style. It replaces the imagery decoders, the timing predictor, `SelfModel.predict` and the hand-written curiosity terms. Ensemble disagreement becomes the intrinsic reward. Agency becomes the change in predicted outcome across alternative actions, scored against simulator counterfactuals. | One component with a loss and an evaluation (k-step prediction error, agency AUROC) replaces several components that have neither. | [15 §5](15-horizon-2-spec.md#5-world-model-agency-and-intrinsic-reward); 02 C5, C8; D-12 |
| R6 | **One calibrated uncertainty service**, using ensembles or MC dropout with temperature scaling. Its consumers are the trigger variant, concept novelty, H3 halting and escalation, curiosity and self-assessment. Evaluate it with ECE, misclassification AUROC and risk–coverage, after every task and every growth event. | Uncertainty appears in at least six places in the docs and is never calibrated or evaluated. | 14 §3.2, §4.5 and test 8; 15 §6; 02 C3; D-10 |
| R7 | **Specify the H2 interface before the split is frozen.**<br>• Observations: symbolic grid as primary, RGB render as secondary, mirroring Tracks A and B.<br>• Mission encoder: frozen scaffold text encoder + FiLM.<br>• Learner: imitation from the BabyAI bot, with an optional PPO fine-tune.<br>• Continual learning for policies: Progress & Compress, CLEAR-style replay.<br>• RL growth signals: success-rate plateau, underfit relative to earlier levels.<br>• Say whether task boundaries are given. | Doc 14's signals (validation loss, train accuracy, boundaries) don't exist as such for a policy. Training a GRU mission encoder from scratch would breach PLAN §5. | [15 §2–§3](15-horizon-2-spec.md#2-interface); doc 05 §3.2; D-11 |
| R8 | **Make System 2 slots concrete in H3.** A slot is an open entry in the deliberation search frontier (a sub-goal or partial program), prioritized by verifier value. The {2, 4, 7} ablation then means frontier width at equal compute. Verification stays external. | As written, "slots" have no operational meaning in a propose → verify → revise loop. Intrinsic self-correction without external feedback does not reliably improve reasoning (Huang et al., ICLR 2024). | [16 §3](16-horizon-3-spec.md#3-system-2-slots-as-the-search-frontier); PLAN §2; 02 §0.2, C10 |
| R9 | **H3 evaluation hygiene.**<br>• Report contamination-robust variants (GSM-Symbolic) next to GSM8K.<br>• The single-pass baseline also executes programs, so the delta measures "revise", not "runs Python".<br>• The program executor is a tool, so the TRK-17 sandbox and red-team suite cover it.<br>• ToM is measured as a delta over the scaffold alone. | Without these, H3 numbers measure SmolLM2 and the Python interpreter, not E-Brain. | [16 §4–§5](16-horizon-3-spec.md#4-baselines); PLAN §3.3; doc 05 §3.3 |
| R10 | **Rewrite doc 02 §1–§7 as about 10 component cards** (I/O, loss, horizon, evaluation), and move the rest to an "unscheduled" appendix, as Appendix A did for neurons. See §11.3. | Only 3 of the 27 components define I/O, loss, horizon and evaluation, and all three come from doc 14. | [02 §1–§2](02-architecture.md#1-component-cards); [Appendix B](appendix-b-architecture-v1-unscheduled.md) (TRK-25) |
| R11 | **No deployment and no learning from real users before Gate 3** plus the TRK-17/TRK-23 safety review. | The plan has no user-facing deployment, but the old docs describe online learning from users. Making this a decision on record closes that path. | PLAN D-16 |
| R12 | **Process.**<br>• Turn each PLAN §5 row into a short decision record.<br>• Extend the docs lint to flag gate thresholds or month numbers outside PLAN.md.<br>• Move superseded docs to `docs/archive/` once their replacements land.<br>• Tag configs before each gate sweep (pre-registration).<br>• Publish each gate report, No-Go included, as a preprint or workshop paper. | It keeps the single source of truth honest, and it gets outside review at each gate. | [DECISIONS.md](DECISIONS.md); PLAN §3 (pre-registration), §8; `tests/test_single_source.py`; [docs/archive/](archive/README.md) |
| R13 | **Write a risk register with explicit priors.** Estimate the chance of passing each gate and when to stop, and get an advisor's sign-off on each pre-registration. | The project currently depends on one person, and Gate 1's controller claim has a real chance of failing (§6 already plans for it). | PLAN §6.2 |

### 11.3 Doc 02 §1–§7: component triage and code-level defects

Verdicts from the second pass, condensed:

- **Keep (with the doc 14 version):**
  - dynamic core;
  - growth controller (it needs a redesign for H2);
  - performance monitor, which becomes the CL harness;
  - vision encoder, as the scaffold;
  - language generator, as the scaffold plus adapters, counting only deltas.
- **Redesign:**
  - working memory: an H2 recurrent policy state, plus the H3 slots (R8);
  - multi-stage reasoning: the H3 outer loop only;
  - long-term memory: a budgeted replay store plus concept grounding IDs (R3);
  - uncertainty estimator (R6);
  - concept graph (TRK-19);
  - decision making: a discrete policy head with continual learning for policies (R7);
  - reward: narrowed to the H2 environment reward plus one disagreement bonus (TRK-18);
  - self-identity: an agency detector plus capability calibration;
  - entity tracking: H2 object tracks, or slots.
- **Defer (beyond H3):** audio encoder, motor control, Theory of Mind module, multi-agent coordinator, peer communication protocol.
- **Cut:**
  - unified embedding space (no gate needs it);
  - concurrent thought system (it duplicates System 2 slots);
  - timing system;
  - sensory-grounded thought system (replaced by R5);
  - correction mechanism (merged into H3 revise, replay and TRK-23 quarantine);
  - person-perspective system (the scaffold handles deixis).

Code-level defects found in §1–§7. They support cutting or rewriting rather than patching. Line numbers are as of November 25, 2026, and several were spot-checked.

| Location (02:) | Defect |
|---|---|
| 3103–3109 | `_activate_hierarchy` aliases `concept.parent_concepts` and extends it while looping over it. That never terminates, and it corrupts the graph. |
| 1747–1766 | "Sleep" prunes every weight below 0.01 and rescales every `.weight` in `network.modules()` to unit norm, including the scaffold and normalization layers. That would destroy the trained network, break the EWC anchors, and fail the function-preservation test. |
| 4053–4058 | Pronoun grounding maps any speaker's "I" to E-Brain itself. |
| 3962–3965 | `infer_belief` returns the first matching observation, regardless of recency. |
| 3771, 3854 | Identity vectors are `torch.randn`. The self-identity vector trains on nothing, and entity re-identification can never match, so every observation registers a new entity. |
| 630–640, 788–796, 1085–1092 | Completed thoughts are kept in `self.thoughts` and count against capacity. The attention controller never selects suspended thoughts, so they never resume. |
| 3693–3697, 3700–3724 | The stack table and config block still say PyTorch Lightning, a 768-d 12-head core, A-GEM and a 10k buffer. That matches neither H1 track. |
| 1862–1892, 1514–1540, 1900, 1909, 4276–4339 | Spiking/STDP is wired into the timing system. Phases advance on elapsed seconds and carry calendar months. Consolidation starts only in Phase 4, although H1 consolidates at every task boundary. |
| 274, 4211, 4370, 4401–4415 | Working memory is "expandable". `send_to_peer` sits outside governance. There is a reward-weighted "SocialAlignmentReward". "Direct embedding exchange" between independently grown peers is cross-architecture transfer, which PLAN §5 makes research-only. |

These became trackers TRK-25 to TRK-29 in [§10](#10-proposed-action-items). Doc 02 §1–§7 now lives in [Appendix B](appendix-b-architecture-v1-unscheduled.md), and the `02:NNNN` line numbers above refer to doc 02 before that move. Line N is line N − 115 of Appendix B.
