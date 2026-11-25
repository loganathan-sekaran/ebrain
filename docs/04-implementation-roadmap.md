# E-Brain Implementation Roadmap

## Overview

> **Schedule, gates, benchmarks, budget and team live in [PLAN.md](PLAN.md) only.** This roadmap lists the *work items* for each stage and maps stages to horizons. Stage headings deliberately carry no month numbers.
>
> Horizon 1 is specified in full in [14-horizon-1-spec.md](14-horizon-1-spec.md). That spec replaces the Stage 1 detail that used to live here.

| Stage | Horizon | Content |
|---|---|---|
| 0 | H1 (before Month 1) | Repo hygiene, CI, test skeleton (TRK-15) |
| 1 | H1 | Growth + continual learning, Gate 1 |
| 2 | H2 | Grounded perception, agency |
| 3 | H2 | Concept graph, compositional generalization, Gate 2 |
| 4 | H3 | Deliberation, governed tools |
| 5 | H3 → Beyond | Adapter clones (H3, Gate 3). Expertise, ecosystem and concept transfer are unscheduled. |

> **Note on Stages 2–5.** Their task lists predate the November 2026 review. Some items are out of scope for the gate they sit under until a horizon spec adopts them: Atari, speech, circadian/sleep, RAVEN, chess, the ecosystem hub. Stage 3's language, audio and unified-embedding items (3.1–3.3) belong to H3 under [PLAN.md §2](PLAN.md#developmental-phases-mapped-to-horizons), not H2. Each horizon spec, written before its horizon starts (TRK-19, TRK-17), decides what stays. Drafts exist for H2 ([15](15-horizon-2-spec.md)) and H3 ([16](16-horizon-3-spec.md)); where a stage item conflicts with them, the spec wins. Components are scheduled only through the cards in [02 §1](02-architecture.md#1-component-cards).

---

## Stage 0: Project Setup — [Horizon 1]

### Goals
- Set up development environment
- Create project structure
- Establish workflows and tooling

### Tasks

#### Infrastructure
- [ ] Set up Git repository with proper .gitignore
- [ ] Create virtual environment (Python 3.12+)
- [ ] Install core dependencies (PyTorch, transformers, etc.)
- [ ] Set up experiment tracking (Weights & Biases)
- [ ] Configure GPU/cloud computing resources

#### Project Structure
```
ebrain/
├── docs/                  # Documentation (current)
├── src/ebrain/           # Source code
│   ├── core/            # Core components
│   ├── models/          # Neural architectures
│   ├── trainers/        # Training loops
│   ├── data/            # Data loaders
│   └── utils/           # Utilities
├── experiments/         # Experiment configs
├── tests/              # Unit tests
├── scripts/            # Training/evaluation scripts
├── notebooks/          # Jupyter notebooks for exploration
├── checkpoints/        # Model checkpoints
└── results/            # Experimental results
```

#### Development Tools
- [ ] Set up formatting and linting (Ruff)
- [ ] Set up type checking (mypy)
- [ ] Set up testing framework (pytest)
- [ ] Create CI pipeline (GitHub Actions) and pre-commit hooks
- [ ] Set up documentation generation (Sphinx)

### Deliverables
- ✅ Working development environment
- ✅ Project skeleton with placeholder modules
- ✅ README with setup instructions
- ✅ First commit and main branch protection

---

## Stage 1: Growth + Continual Learning — [Horizon 1]

Fully specified in [14-horizon-1-spec.md](14-horizon-1-spec.md):

- hypothesis and novelty (§1);
- tracks (§2);
- protocol, metrics, baselines and ablations (§3);
- growth mechanism (§4);
- implementation constraints and required tests (§5);
- compute (§6);
- month-by-month plan (§7);
- Gate 1 report template (§8).

Gate 1 itself is defined in [PLAN.md §3.1](PLAN.md#31-gate-1-end-of-month-3).

### Work items
- [ ] CL harness: streams, metrics, JSONL logging, Hydra config
- [ ] Baselines from [14 §3.3](14-horizon-1-spec.md#33-baselines), reproduced against published numbers. The pre-allocated and width-matched static controls come after the grown runs.
- [ ] `GrowableNetwork.grow()` with function-preserving residual blocks. Never override `nn.Module.add_module`.
- [ ] `GrowthController` (plateau ∧ (underfit ∨ interference), warm-up, cooldown, cap) as a pure, unit-tested function
- [ ] Optimizer param-group handling on growth; online EWC that tolerates new parameters
- [ ] Full sweep and ablations (random-time and boundary growth), seeds per [PLAN.md §3](PLAN.md#decision-rule-all-gates), Gate 1 report

---

## Stage 2: MVP - Grounded Perception & Agency — [Horizon 2: Grounded Agency]

### Goals
- Build complete pipeline for one modality (vision)
- Implement all core components
- Achieve developmental progression through Phase 1-2

### Tasks

#### 2.1 Complete Vision Pipeline
- [ ] Implement vision encoder (CNN + ViT)
- [ ] Add data augmentation pipeline
- [ ] Create vision-specific evaluation suite
- [ ] Integrate with growth mechanism

#### 2.2 Add Memory Systems
- [ ] Implement working memory (attention-based)
- [ ] Build long-term memory (vector database)
- [ ] Add episodic buffer for experience replay
- [ ] Create memory retrieval mechanisms
- [ ] **Basic multi-stage reasoning scaffold**
  - [ ] Simple 2-stage processing (initial + refinement)
  - [ ] Foundation for later expansion to 5 stages
- [ ] **Atomic concept learning foundation**
  - [ ] Concept graph structure (NetworkX)
  - [ ] Atomic concept detection (colors, shapes, simple patterns)
  - [ ] Concept embedding storage
- [ ] **Self-identity initialization**
  - [ ] Create SelfIdentitySystem class
  - [ ] Initialize BodySchema (sensor/actuator mapping)
  - [ ] Create persistent identity_vector (512-dim)
  - [ ] Setup autobiographical memory storage

#### 2.3 Build Meta-Learning Components
- [ ] Performance monitoring system
- [ ] Uncertainty estimation (MC Dropout, ensembles)
- [ ] Automated evaluation pipeline
- [ ] Logging and visualization dashboards
- [ ] **Developmental reward system foundation**
  - [ ] Implement PredictionReward module
  - [ ] Implement CuriosityReward module
  - [ ] Phase 1 reward weighting (accuracy + novelty)
  - [ ] Reward logging and visualization

#### 2.4 Implement Phase 1 Curriculum
- [ ] Collect/prepare datasets (MNIST, CIFAR, simple objects)
- [ ] Create curriculum scheduler
- [ ] Implement difficulty progression
- [ ] Add success criteria checking
- [ ] **Learn atomic visual concepts**
  - [ ] Colors (10-15 basic colors)
  - [ ] Shapes (circle, square, triangle, etc.)
  - [ ] Textures (smooth, rough, striped)
  - [ ] Basic edges and corners
- [ ] **Reward-driven learning**
  - [ ] Track prediction accuracy rewards
  - [ ] Bonus for novel pattern discovery
  - [ ] Self-supervised learning loop
- [ ] **Agency detection (Phase 1)**
  - [ ] Implement AgencyDetector class
  - [ ] Motor babbling experiments (random actions)
  - [ ] Learn "I caused this" vs "external event"
  - [ ] Track self-caused action outcomes
- [ ] **Internal timing system basics (Phase 1)**
  - [ ] Implement MillisecondTimer class
  - [ ] Implement IntervalTimer class
  - [ ] Basic timestamp tracking for observations
  - [ ] No prediction yet, just recording
- [ ] **Sensory-grounded thoughts (Phase 1)**
  - [ ] Implement SensoryGroundedThoughtSystem class
  - [ ] Implement SensoryGroundingDatabase class
  - [ ] Simple sensory associations (visual pattern → label)
  - [ ] Store concept groundings (basic)
  - [ ] No imagery generation yet, just association learning

#### 2.5 Add Action Generation
- [ ] Simple decision-making module
- [ ] Policy network for grid world
- [ ] Integration with RL environments (Gym)
- [ ] Reward processing and learning
- [ ] **Phase 2 reward system**
  - [ ] Implement ExplorationReward module
  - [ ] Implement CompetenceGrowthReward module
  - [ ] Task success + exploration + diversity weighting
  - [ ] State visit tracking for exploration bonus

#### 2.6 Phase 2 Development
- [ ] BabyAI environment integration
- [ ] Simple Atari games (Pong, Breakout)
- [ ] Multi-step planning module
- [ ] Strategy formation tracking
- [ ] **Exploration-driven learning**
  - [ ] Reward new state discovery
  - [ ] Reward action diversity
  - [ ] Balance exploitation vs exploration
- [ ] **Self-other distinction (Phase 2)**
  - [ ] Implement EntityTracker class
  - [ ] Register "SELF" as first entity
  - [ ] Track objects as separate entities
  - [ ] Distinguish self-caused from external events
  - [ ] Basic perspective: "my position" vs "object position"
- [ ] Multi-step planning module
- [ ] Strategy formation tracking

#### 2.7 Testing & Refinement
- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Bug fixes and stability improvements
- [ ] **Concurrent thought system (Phase 2)**
  - [ ] Implement ThoughtStream class
  - [ ] Implement ConcurrentThoughtSystem with 2 thought limit
  - [ ] Implement basic AttentionController
  - [ ] Test dual thought processing (navigate + remember)
  - [ ] Context switching mechanism
- [ ] **Timing system expansion (Phase 2)**
  - [ ] Implement TemporalPredictor class
  - [ ] Implement ActionScheduler class
  - [ ] Learn temporal patterns (action → reward delay)
  - [ ] Test action timing accuracy
  - [ ] Context-dependent timing adjustments

### Success Criteria
- ✅ Complete Phase 1 successfully (>85% on vision tasks)
- ✅ Complete Phase 2 successfully (solve BabyAI tasks)
- ✅ Demonstrate growth across 5+ tasks
- ✅ Zero catastrophic forgetting
- ✅ System runs stably for extended training

### Deliverables
- Complete single-modality E-Brain
- Trained checkpoint through Phase 2
- Evaluation report with metrics
- Technical documentation

---

## Stage 3: Multi-Modal Integration & Concept Graphs — [Horizon 2: Concept Hierarchy]

### Goals
- Add scaffolded language and audio perception
- Implement hierarchical concept composition (Level 0 to 3)
- Achieve Phase 3 (symbol grounding and social agency)

### Tasks

#### 3.1 Scaffolded Language Processing
- [ ] Integrate lightweight pre-trained language backbone (e.g. SmolLM-135M / TinyLlama) as linguistic encoder
- [ ] Tokenizer & embedding projection layers to E-Brain unified space
- [ ] Language decoding head for generation
- [ ] Text evaluation metrics on grounded QA
- [ ] **Sensory-grounded thoughts expansion (Phase 2)**
  - [ ] Implement VisualImageryGenerator class (basic)
  - [ ] Implement TactilePredictor class (basic)
  - [ ] Basic sensory grounding: object → visual + tactile
  - [ ] Test grounding with common objects (cup, ball, etc.)
  - [ ] No mental imagery yet, just feature association

#### 3.2 Add Audio Processing
- [ ] Audio encoder (spectrogram + transformer)
- [ ] Speech recognition pipeline
- [ ] Audio-visual sync detection
- [ ] Audio evaluation metrics

#### 3.3 Unified Embedding Space
- [ ] Cross-modal projection heads
- [ ] Contrastive learning for alignment
- [ ] Multi-modal fusion layers
- [ ] Cross-modal attention mechanisms

#### 3.4 Phase 3 Implementation
- [ ] BabyAI language grounding
- [ ] Word-object association tasks
- [ ] Grammar induction experiments
- [ ] Simple question answering
- [ ] **Basic concept composition**
  - [ ] Compose atomic concepts into basic objects (red + round → ball)
  - [ ] Level-1 concept formation
  - [ ] Concept composition engine implementation
- [ ] **Phase 3 reward system**
  - [ ] Implement CommunicationReward module
  - [ ] Implement HumanFeedbackProcessor
  - [ ] Human validation integration (praise/criticism)
  - [ ] Concept learning rate tracking
- [ ] **Person recognition system (Phase 3)**
  - [ ] Implement PersonPerspectiveSystem class
  - [ ] Pronoun grounding ("I", "you", "he/she/they")
  - [ ] Entity identification from sensory input
  - [ ] Register humans as separate entities
  - [ ] Track current addressee in conversation

#### 3.5 Multi-Modal Curriculum
- [ ] Vision + language tasks
- [ ] Audio + language tasks
- [ ] Vision + audio + language integration
- [ ] Cross-modal transfer experiments
- [ ] **Cross-modal concept learning**
  - [ ] Same concept across modalities (word "dog" + image of dog + bark sound)
  - [ ] Multi-modal concept embeddings
- [ ] **Communication-driven learning**
  - [ ] Reward successful human understanding
  - [ ] Learn from human corrections
  - [ ] Symbol grounding accuracy rewards

#### 3.6 Symbol Grounding
- [ ] Word-to-concept mapping
- [ ] Compositional understanding (adjective + noun)
- [ ] Instruction following
- [ ] **Hierarchical concept learning**
  - [ ] Level-2 concepts: Objects from parts
  - [ ] Concept correlation learning
  - [ ] Parent-child concept relationships
- [ ] **Human utility reward introduction**
  - [ ] Track how well E-Brain helps humans
  - [ ] Begin transition to utility-driven motivation
- [ ] Simple dialogue capability
- [ ] **Theory of Mind basics (Phase 3)**
  - [ ] Implement TheoryOfMindSystem class
  - [ ] Track beliefs per entity (what does X know?)
  - [ ] Basic false belief understanding
  - [ ] Goal inference (what does X want?)
  - [ ] Perspective taking (simulate X's view)
- [ ] **Concurrent thought expansion (Phase 3)**
  - [ ] Expand to 4 concurrent thoughts
  - [ ] Implement SharedInsightMemory
  - [ ] Cross-pollination mechanism (basic)
  - [ ] Test: read + integrate + predict simultaneously
  - [ ] Reduce context switch cost to 0.15
- [ ] **Sleep/consolidation system (Phase 3)**
  - [ ] Implement SleepConsolidationSystem class
  - [ ] Implement DevelopmentalTimer class
  - [ ] Memory consolidation (episodic → semantic)
  - [ ] Experience replay during sleep
- [ ] **Sensory-grounded thoughts (Phase 3)**
  - [ ] Implement AuditorySimulator class
  - [ ] Inner speech capability (think_in_words)
  - [ ] Mental imagery for problem solving (think_visually)
  - [ ] Action imagination (imagine_action)
  - [ ] Test: reason_with_imagery for visual problems
  - [ ] Homeostatic scaling
  - [ ] Test: learning improvement after sleep cycles

### Success Criteria
- ✅ Complete Phase 3 (language acquisition)
- ✅ Understand 1000+ words
- ✅ Follow complex instructions
- ✅ Demonstrate cross-modal transfer
- ✅ Generate grammatically correct sentences

### Deliverables
- Multi-modal E-Brain system
- Language-capable checkpoint
- Cross-modal evaluation suite
- Multi-modal learning paper

---

## Stage 4: Deliberative Reasoning & Tool Use — [Horizon 3: Reasoning & Tools]

### Goals
- Implement Phase 4 capabilities (deliberative multi-stage reasoning & tool orchestration)
- Achieve cross-domain transfer learning and meta-cognition
- Develop sandboxed subprocess & API tool execution abilities

### Tasks

#### 4.1 Reasoning Module
- [ ] Enhanced transformer for reasoning
- [ ] **Multi-stage reasoning system implementation**
  - [ ] StageProcessor with progressive depth
  - [ ] DepthController for adaptive reasoning
  - [ ] VerificationModule for self-checking
  - [ ] Reasoning trace logging and visualization
- [ ] Chain-of-thought implementation
- [ ] Logical inference engine
- [ ] Causal reasoning components

#### 4.2 Concept Learning
- [ ] **Advanced hierarchical concept system**
  - [ ] Level-3 concepts: Categories and abstract groupings
  - [ ] Multiple composition types (and/or/sequence/spatial/functional)
  - [ ] Concept explanation generation
  - [ ] Compositional generalization testing
- [ ] **Multi-stage concept understanding**
  - [ ] Progressive depth learning (surface → deep)
  - [ ] Iterative refinement for ambiguous concepts
  - [ ] Integration with long-term memory
- [ ] Prototype-based learning
- [ ] Analogy making system (concept similarity and transfer)
- [ ] Abstract pattern completion
- [ ] **Phase 4 reward system**
  - [ ] Implement ProblemSolvingReward module
  - [ ] Implement TransferLearningReward module
  - [ ] Solution elegance rewards (efficiency)
  - [ ] Cross-domain transfer bonuses
  - [ ] Increase human utility weight
- [ ] **Multi-agent coordination (Phase 4)**
  - [ ] Implement MultiAgentCoordinator class
  - [ ] Track multiple entities simultaneously
  - [ ] Group conversation handling
  - [ ] Attention allocation in multi-entity scenes
  - [ ] Relationship graph maintenance

#### 4.3 Phase 4 Curriculum
- [ ] RAVEN's matrices (with multi-stage reasoning)
- [ ] CLEVR reasoning (decomposition mode)
- [ ] bAbI tasks (chain-of-thought mode)
- [ ] Mathematical reasoning (step-by-step verification)
- [ ] **Recursive mental state reasoning (Phase 4)**
  - [ ] Implement recursive belief tracking (X thinks Y believes Z)
  - [ ] Goal inference from observed behavior
  - [ ] Action prediction based on mental models
  - [ ] Strategic communication decisions

#### 4.4 Transfer Learning
- [ ] Few-shot learning meta-module
- [ ] Cross-domain transfer experiments
- [ ] Knowledge distillation
- [ ] Domain adaptation techniques
- [ ] **E-Brain peer collaboration (Phase 4)**
  - [ ] Implement E-Brain-to-E-Brain communication protocol
  - [ ] Identity exchange and verification
  - [ ] Capability sharing
  - [ ] Task division and coordination
  - [ ] Multi-E-Brain project execution
- [ ] **Full working memory (Phase 4)**
  - [ ] Expand to 7 concurrent thoughts
  - [ ] Advanced attention strategies (stuck detection, priority boosting)
  - [ ] Background processing for difficult problems
  - [ ] Enhanced cross-pollination (creative insights)
  - [ ] Test: complex problem solving with multiple approaches
- [ ] **Circadian rhythm system (Phase 4)**
  - [ ] Implement CircadianClock class
  - [ ] 24-hour cycle simulation
  - [ ] Active vs rest period differentiation
  - [ ] Strategic sleep scheduling
  - [ ] Learning efficiency tracking by time of day
- [ ] **Rich sensory-grounded thoughts (Phase 4)**
  - [ ] Implement MultimodalBinder class
  - [ ] Multimodal reasoning (visual + auditory + tactile)
  - [ ] Rich concept grounding (multiple sensory dimensions)
  - [ ] Complex action simulation
  - [ ] Abstract concepts via sensory metaphor
  - [ ] Test: multimodal problem solving (e.g., "safely pour hot water")

#### 4.5 Meta-Cognition
- [ ] Confidence calibration
- [ ] Explanation generation
- [ ] Knowledge gap identification
- [ ] Self-assessment mechanisms

### Success Criteria
- ✅ Complete Phase 4 (abstract reasoning)
- ✅ >70% on RAVEN's matrices
- ✅ Demonstrate transfer learning
- ✅ Generate explanations for decisions
- ✅ Identify knowledge boundaries

### Deliverables
- Reasoning-capable E-Brain
- Transfer learning benchmarks
- Meta-cognitive evaluation suite
- Research paper on developmental AI

---

## Stage 5: Expertise, Model Cloning & Knowledge Transfer — [Horizon 3: Transfer & Ecosystem]

### Goals
- Implement Phase 5 capabilities (domain specialization and self-curriculum)
- Implement Model Cloning infrastructure for specialized variants
- Enable Teacher-Student Knowledge Transfer protocol and collaborative ecosystem

### Tasks

#### 5.1 Domain Specialization
- [ ] Choose 2-3 target domains (e.g., chess, coding, Q&A)
- [ ] Implement domain-specific modules
- [ ] Deep training in each domain
- [ ] Competition-level evaluation

#### 5.2 Mixture of Experts
- [ ] MoE architecture integration
- [ ] Expert routing mechanisms
- [ ] Load balancing
- [ ] Expert specialization tracking

#### 5.3 Self-Curriculum Generation
- [ ] Automated difficulty assessment
- [ ] Task generation for weak areas
- [ ] Exploration strategies
- [ ] Intrinsic motivation mechanisms

#### 5.4 Teaching Capability
- [ ] Explanation generation refinement
- [ ] Pedagogical strategies
- [ ] Student modeling
- [ ] Adaptive teaching
- [ ] **Mature social identity (Phase 5)**
  - [ ] Implement role-based interaction system
  - [ ] Autobiographical memory narrative
  - [ ] Purpose and values definition
  - [ ] Social norm understanding
  - [ ] Teaching E-Brain students capability
- [ ] **Expert concurrent thought (Phase 5)**
  - [ ] Optimize attention controller (5% context switch cost)
  - [ ] Large insight database (10k insights)
  - [ ] Graceful interrupt handling (suspend/resume all thoughts)
  - [ ] Background creativity (generate questions while researching)
  - [ ] Test: complex research task with 5+ concurrent threads
- [ ] **Master timing system (Phase 5)**
  - [ ] Full circadian integration (active 6am-10pm, rest 10pm-6am)
  - [ ] Context-aware sleep scheduling
  - [ ] Multi-scale temporal prediction (millisecond to month)
  - [ ] Adaptive action timing (<10ms accuracy)
  - [ ] Learned temporal rhythms (user patterns, resource availability)
  - [ ] Smart consolidation scheduling (optimize learning efficiency)
- [ ] **Expert sensory-grounded thoughts (Phase 5)**
  - [ ] Expert mental imagery (generate novel scenes never seen)
  - [ ] Complex internal dialogue (multi-perspective reasoning)
  - [ ] Advanced action simulation (predict complex consequences)
  - [ ] Abstract reasoning via sensory metaphor (e.g., ethical concepts)
  - [ ] Rich semantic networks (abstract-to-concrete concept grounding)
  - [ ] Multimodal creative thinking (design novel solutions)
  - [ ] Test: creative problem solving, metaphorical reasoning
- [ ] **Social identity maturation (Phase 5)**
  - [ ] Teacher/peer/student/user relationship handling
  - [ ] Teach other E-Brains (student role for them)
  - [ ] Social norm understanding
  - [ ] Turn-taking and group conversation etiquette
  - [ ] Autobiographical memory reflection
  - [ ] Purpose and values representation
  - [ ] Long-term relationship maintenance

#### 5.5 Creative Problem Solving
- [ ] Novel solution generation
- [ ] Cross-domain knowledge combination
- [ ] Creative exploration mechanisms
- [ ] Evaluation of creativity

#### 5.6 Model Cloning Infrastructure
- [ ] Implement model cloning system
- [ ] Create full clone, partial clone, and adapter-based strategies
- [ ] Build domain-specific initialization framework
- [ ] Test cloning with first domain (e.g., chess or coding)

#### 5.7 Knowledge Transfer Protocol
- [ ] Implement concept extraction from teacher models
- [ ] Build knowledge package format (embeddings, reasoning patterns, examples)
- [ ] Create transfer protocol (alignment, integration, verification)
- [ ] Test teacher-student transfer on simple concepts

#### 5.8 Multi-Model Ecosystem
- [ ] Build shared knowledge hub
- [ ] Enable concept contribution and retrieval
- [ ] Implement curriculum builder for multi-concept transfer
- [ ] Create ecosystem monitoring and analytics

### Success Criteria
- ✅ Expert-level in 2+ domains
- ✅ Beat human baselines
- ✅ Self-improve without supervision
- ✅ Generate creative solutions
- ✅ Teach concepts effectively
- ✅ Successfully clone and specialize base model
- ✅ Transfer concepts between models with >80% success rate
- ✅ Knowledge hub with 50+ shareable concepts

### Deliverables
- Complete E-Brain system
- Expert-level checkpoints
- Self-learning demonstration
- Model cloning and transfer system
- Knowledge hub infrastructure
- Final research publication

---

## Parallel Workstreams

### Throughout All Stages

#### Research & Literature Review
- Monitor latest papers in continual learning, NAS, developmental AI
- Participate in relevant conferences
- Collaborate with research community

#### Documentation
- Maintain code documentation
- Update architecture docs as system evolves
- Write blog posts about progress
- Create tutorial notebooks

#### Evaluation & Benchmarking
- Regular evaluation on standard benchmarks
- Track metrics across all developmental phases
- Compare with state-of-the-art systems
- Publish benchmark results

#### Optimization & Efficiency
- Profile and optimize bottlenecks
- Reduce memory footprint
- Improve training speed
- Explore distributed training

---

## Risk Management

### High-Risk Items & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Catastrophic forgetting | High | Critical | Multiple CL algorithms, extensive testing |
| Computational costs | High | High | Start small, cloud resources, efficient architectures |
| Growth mechanism failure | Medium | High | Extensive PoC phase, multiple strategies |
| Phase progression failure | Medium | High | Clear success criteria, fallback plans |
| Scope creep | Medium | Medium | Strict phase boundaries, MVP focus |
| Knowledge transfer compatibility | Medium | Medium | Standardized concept packages, alignment layers |
| Clone specialization quality | Medium | Medium | Careful base model selection, validation protocols |

---

## Resources, Success Metrics and Gates

Compute, storage, team, gate criteria and adaptation triggers are defined only in [PLAN.md](PLAN.md):

- §3: horizons, gates and benchmarks;
- §4: budget and team;
- §6: adaptation triggers.

Metric definitions (ACC, BWT, FWT, forgetting) are in [14-horizon-1-spec.md §3.2](14-horizon-1-spec.md#32-metrics).

---

## Next Steps

See [PLAN.md §7](PLAN.md#7-next-steps).

---

*Roadmap Version: 2.0*  
*Last Updated: November 25, 2026*
