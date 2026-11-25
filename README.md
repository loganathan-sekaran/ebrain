# E-Brain: Developmental AI with Dynamic Growth

[![Status](https://img.shields.io/badge/status-planning-blue)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

> A developmental artificial intelligence system that grows and learns like a human brain, starting from minimal capabilities and progressing through structured phases to achieve expert-level performance across multiple domains.

## 🧠 Overview

E-Brain is an ambitious research project to create an AI system that:

- **Starts Simple**: Begins with minimal capabilities, like a newborn
- **Grows Dynamically**: Expands its neural architecture based on learning needs
- **Learns Progressively**: Follows developmental phases mimicking human cognitive growth
- **Masters Multiple Modalities**: Processes vision, language, audio, and more
- **Self-Corrects**: Learns from feedback and teaching
- **Specializes via Cloning**: Creates domain-specific experts from foundation model
- **Transfers Knowledge**: Enables teacher-student concept transfer between models
- **Forms Ecosystems**: Builds collaborative networks of specialized AI models

## 🎯 Key Innovations

> **Vision vs. plan.** The sections below describe the long-term vision. What is actually scheduled is the three horizons in [docs/PLAN.md](docs/PLAN.md): growth + continual learning, then grounded concepts + agency, then deliberation, governed tools and adapter clones. Everything else here is unscheduled until Gate 3. Where a section below conflicts with PLAN.md's decisions on record, PLAN.md wins.

### 1. Dynamic Architecture Growth
Unlike fixed neural networks, E-Brain grows new modules and layers when needed, similar to how the brain forms new neural connections.

### 2. Developmental Phases
Learning follows human-inspired stages:
- **Phase 0**: Initialization (Birth) - Basic sensory capabilities
- **Phase 1**: Sensory Learning (0-6mo) - Pattern recognition
- **Phase 2**: Motor Control (6-18mo) - Action-consequence learning
- **Phase 3**: Language Acquisition (18mo-3yr) - Symbol grounding, communication
- **Phase 4**: Abstract Reasoning (3-7yr) - Logical thinking, transfer learning
- **Phase 5**: Expertise (7yr+) - Domain mastery, self-directed learning

### 3. Concept-Driven Learning
Hierarchical concept formation from atomic to expert-level concepts:
- **Atomic Concepts** (Level 0): Colors, shapes, edges, tones (like pixels/phonemes)
- **Basic Concepts** (Level 1): Object parts, simple patterns (combining atomic features)
- **Intermediate Concepts** (Level 2): Complete objects, words (composing basic concepts)
- **Abstract Concepts** (Level 3): Categories, relationships (generalizing from examples)
- **Expert Concepts** (Level 4): Domain theories, meta-concepts (high-level abstractions)

Concepts compose through multiple rules: AND (all features), OR (alternatives), SEQUENCE (order matters), SPATIAL (arrangement), FUNCTIONAL (relationships). Enables compositional generalization—understanding "red car" even if never seen one, by combining existing "red" and "car" concepts.

### 4. Human-Like Learning + Machine-Like Computation

**E-Brain's Hybrid Philosophy**: Learn *how* humans think, compute *like* a machine.

**Human-Like Aspects** (How we learn):
- 🧒 Developmental stages (infant → expert)
- 🎨 Sensory grounding (think in images, sounds, feelings)
- 🤔 Curiosity-driven exploration
- 👥 Social cognition and Theory of Mind
- 💭 Abstract reasoning and creativity

**Machine-Like Advantages** (What we can do):
- ⚡ **Large-scale retrieval** (System 1): associative memory bounded by storage, not by a 7±2 limit
- 🔄 **Parallel search** (System 1): batched rollouts and hypothesis evaluation on GPU
- 🎯 **Bounded deliberation** (System 2): a small, configurable number of active goals that keeps reasoning focused
- 🛠️ **Governed tool use**: read-only tools are open; side-effecting tools run only through a sandbox with approval gates and an audit log

**Result**: An AI that learns the way humans do, uses machine-scale memory and search, and acts in the world only through governed, auditable tools. See [docs/02-architecture.md §0.2](docs/02-architecture.md#02-dual-process-cognitive-architecture-system-1-vs-system-2).

### 5. Continual Learning
Prevents catastrophic forgetting through multiple strategies (EWC, A-GEM, Experience Replay, Progressive Networks).

### 6. Multi-Stage Reasoning
Enables iterative, deliberate thinking for complex tasks through 5-stage progressive reasoning:
- **Stage 0**: Surface pattern recognition (fast path)
- **Stage 1**: Relationship extraction and connections
- **Stage 2**: Deep analysis and abstraction
- **Stage 3**: Integration with prior knowledge
- **Stage 4**: Verification and refinement

Depth adapts based on task complexity and confidence levels, mimicking human thought processes.

### 7. Model Cloning
Once trained, the base model can be cloned to create specialized variants for different domains (medical, coding, games, languages) without training from scratch.

### 8. Knowledge Transfer
Models can teach each other specific concepts through a teacher-student paradigm, transferring reasoning patterns and strategies, not just parameters.

### 9. Intrinsic Motivation & Developmental Rewards
E-Brain's learning evolves from external rewards to intrinsic motivation, mimicking human development:
- **Phase 1 (Infant)**: Prediction accuracy + curiosity (self-supervised)
- **Phase 2 (Toddler)**: Exploration + task success + competence growth
- **Phase 3 (Child)**: Communication success + human feedback (praise/criticism)
- **Phase 4 (Student)**: Problem-solving + transfer learning + mastery
- **Phase 5 (Adult)**: Human utility + continuous improvement, always *within* alignment constraints

**No emotions required**—pure information-theoretic rewards (curiosity = uncertainty reduction) and utility-based goals (helping humans). As E-Brain matures, intrinsic motivation (curiosity, mastery) takes a larger share of the reward, but two things never change. Alignment is a hard constraint that fails closed, not a reward term to trade off. Human corrections keep full weight at every stage ([docs/13 §6.2](docs/13-design-review-2026-11.md#62-reward-design-0241034521)).

### 10. Self-Identity & Social Cognition
E-Brain develops a sense of "I" (self), "You" (others), and "They" (third parties) through embodied experience:

**Developmental Progression:**
- **Phase 1 (Body Schema)**: Maps sensors/actuators—"What I can sense and do"
- **Phase 2 (Agency)**: Distinguishes self-caused from external events—"I vs not-I"
- **Phase 3 (Theory of Mind)**: Tracks others' beliefs and goals—"You know X, but I know Y"
- **Phase 4 (Multi-Agent)**: Coordinates with multiple entities, including E-Brain peers
- **Phase 5 (Social Identity)**: Mature social relationships with purpose and values

**Key Capabilities:**
- **Person Recognition**: Grounds pronouns (I/you/he/she/they) to entities
- **Mental Modeling**: Tracks what each entity knows, wants, and can do
- **Perspective Taking**: Simulates situations from others' viewpoints
- **Multi-Entity Coordination**: Manages group conversations, collaborative tasks
- **E-Brain Collaboration**: Structured protocol for peer-to-peer learning and teamwork
- **Relationship Types**: Teacher (learn from), peer (collaborate), student (teach), user (serve)

Enables **human-like social intelligence** without requiring emotions—pure information-theoretic modeling of mental states and relationships.

### 11. Concurrent Thought Processing
E-Brain processes multiple "thoughts" simultaneously, mimicking the brain's ability to work on several problems at once:

**Key Capabilities:**
- **Bounded Active Goals**: System 2 keeps a small, configurable number of active goal slots (default 4, ablated over 2, 4 and 7)
- **Attention Switching**: Switch focus between thoughts without losing context. There is no artificial switching delay.
- **Background Processing**: Difficult problems "cook" in background while working on easier ones
- **Thought Persistence**: Suspend and resume thoughts without losing context
- **Cross-Pollination**: Insights from one thought help solve others (creative connections!)
- **Priority Management**: Urgent/important tasks can preempt less critical ones

**Examples:**
- Solve math problem while remembering related concepts from previous tasks
- Explore multiple solution approaches in parallel, switch when stuck
- Read research paper: parse current sentence + integrate context + predict next simultaneously
- Handle interruptions gracefully: suspend all thoughts, answer question, resume work

**Slot count:** a hyperparameter of the deliberative controller, chosen by ablation ([docs/PLAN.md §2](docs/PLAN.md#2-terminology)). A growing per-phase slot schedule is only a candidate curriculum variable, not a design fact.

Enables **faster learning and problem-solving** through parallel exploration and creative insight transfer between thought streams.

### 12. Internal Timing and Clock System
E-Brain maintains multi-scale timing mechanisms, mimicking the brain's ability to track time, predict temporal patterns, and time actions accurately:

**Key Capabilities:**
- **Multi-Scale Clocks**: Millisecond precision to circadian rhythms (months)
- **Temporal Prediction**: Learn "when" events will occur, not just "what"
- **Precise Action Timing**: Context-dependent timing adjustments (<20ms accuracy)
- **Consolidation Scheduling**: Memory consolidation triggered by training signals (task boundaries, replay-buffer fill, measured drift), not by the time of day
- **Learned Temporal Patterns**: Discover regularities (user response times, task durations)

**Examples:**
- Predict user responds after ~2 seconds (learned from interaction history)
- Schedule action to execute at precise moment (reward arrives 500ms after action)
- Sleep consolidation: episodic memories → semantic knowledge (like REM sleep)
- Multi-hour prediction: "Training will complete in 3 hours"

Timing is a learned capability (temporal prediction in the environment), not a wall-clock subsystem. The spiking/STDP timing ideas are in the off-critical-path research branch ([Appendix A](docs/appendix-a-bio-inspired-neuron-research-branch.md)).

### 13. **Sensory-Grounded Thoughts** 🎨👂✋

Human thoughts are fundamentally **grounded in sensory experiences** - we think in images, sounds, and feelings, not just abstract symbols. E-Brain implements:

- **Mental Imagery**: Visualize concepts ("picture a red apple"), generate novel scenes
- **Inner Speech**: Auditory thoughts ("hear yourself think"), internal dialogue
- **Tactile Simulation**: Predict how actions feel before executing (motor planning)
- **Multimodal Integration**: Combine multiple senses for robust understanding
- **Concept Grounding**: Link abstract concepts to concrete sensory experiences

**Examples:**
- `think_visually("ocean")` → generates mental image: blue water, waves, horizon
- `think_in_words("What should I say?")` → internal dialogue with auditory representation
- `imagine_action("pick up cup")` → simulates tactile/motor sensations before acting
- `reason_with_imagery("stack blocks")` → uses mental imagery to test solutions
- `ground_concept("justice")` → links abstract concept to spatial metaphors (balance)

**Developmental Progression:**
- **Phase 1**: Simple sensory associations (visual patterns → object labels)
- **Phase 2**: Basic grounding (objects → visual + tactile features)
- **Phase 3**: Inner speech emerges, mental imagery for problem solving
- **Phase 4**: Rich multimodal thoughts, complex sensory simulation
- **Phase 5**: Expert mental imagery, creative thinking, deep metaphorical reasoning

Enables **human-like cognition** through grounded understanding, mental simulation for planning, and richer thought representations.

### 14. **Computational Advantages & Tool Use** ⚡🛠️

E-Brain is **not limited by human biological constraints**. While learning *how* humans think, it computes with superhuman capabilities:

**Computational Superpowers:**
- **Large-scale memory (System 1)**: exact retrieval from a vector store, bounded by storage and a retention policy
- **Parallel search (System 1)**: many batched rollouts evaluated at once on GPU
- **Bounded deliberation (System 2)**: a configurable number of active goal slots (default 4)

**System Tool Integration** (Horizon 3, and only after the tool governance spec, TRK-17, is approved):
- **Subprocess Execution**: Run system commands, execute scripts
- **Code Generation**: Write and execute code dynamically to solve problems
- **API Integration**: Call REST APIs, integrate external services
- **Data Processing**: Leverage pandas, numpy for data manipulation
- **File Operations**: Read, write, manage files and directories
- **Custom Tool Plugins**: Learn and use domain-specific tools

**Tool Learning Progression:**
- **Phase 1**: Basic tool discovery (simple commands)
- **Phase 2**: Tool combination (chain multiple tools)
- **Phase 3**: Strategic tool selection (choose right tool for task)
- **Phase 4**: Tool orchestration (complex multi-tool workflows)
- **Phase 5**: Tool mastery (create custom tools, optimize performance)

**Examples:**
- Generate Python script to analyze data → Execute it → Parse results
- Call GitHub API → Process issues → Generate summary report
- Orchestrate multi-step pipeline: API fetch → data processing → visualization
- Learn new tools through experimentation and build mental models
- Create custom tools optimized for specific problem domains

**Safety boundary:** every side-effecting tool call runs in an isolated sandbox, through a capability-scoped tool registry, behind approval gates, and is recorded in an audit log. See [docs/13-design-review-2026-11.md §6](docs/13-design-review-2026-11.md#6-safety-security-and-alignment).

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Canonical Plan](docs/PLAN.md)** - ⭐ **Single source of truth: horizons, gates, budget, team**
- **[Decision Records](docs/DECISIONS.md)** - Why each decision on record was made
- **[Horizon 1 Spec](docs/14-horizon-1-spec.md)** - Growth + continual learning experiment and mechanism spec
- **[Horizon 2 Spec (draft)](docs/15-horizon-2-spec.md)** - Grounded concepts, world model and agency
- **[Horizon 3 Spec (draft)](docs/16-horizon-3-spec.md)** - Deliberation, tools and adapters
- **[Project Overview](docs/README.md)** - Documentation index and quick start
- **[Feasibility Analysis](docs/01-feasibility-analysis.md)** - Research foundation, technical feasibility, and extended vision
- **[Architecture Design](docs/02-architecture.md)** - Design foundations and component cards
- **[Developmental Phases](docs/03-developmental-phases.md)** - Detailed learning stages and evaluation frameworks
- **[Implementation Roadmap](docs/04-implementation-roadmap.md)** - 24+ month development plan with milestones
- **[Development Strategy](docs/05-development-strategy.md)** - ⭐ **Implementation vs training approach, data requirements, team structure**
- **[Technical Stack](docs/05-technical-stack.md)** - Technology choices, frameworks, and infrastructure
- **[Challenges & Solutions](docs/06-challenges-and-solutions.md)** - Known challenges and mitigation strategies
- **[Model Cloning & Knowledge Transfer](docs/07-model-cloning-and-knowledge-transfer.md)** - Specialization and inter-model learning
- **[Tech Stack Decision Framework](docs/08-tech-stack-decision-framework.md)** - Comprehensive rationale for technology choices
- **[Computational Advantages & Tool Use](docs/10-computational-advantages-and-tool-use.md)** - ⚡ **Superhuman computation + system integration**
- **[Philosophy: Best of Both Worlds](docs/11-philosophy-best-of-both-worlds.md)** - ☯️ **Philosophy: Combining biological and computational intelligence**
- **[Review & Recommendations](docs/12-project-plan-review-and-recommendations.md)** - 📋 **Project plan review (Sept 2026) & tracking matrix**
- **[Design Review (Nov 2026)](docs/13-design-review-2026-11.md)** - Full design review and action items TRK-11 to TRK-29
- **[Appendix A: Bio-Inspired Neuron](docs/appendix-a-bio-inspired-neuron-research-branch.md)** - Spiking/STDP research branch (off the critical path)
- **[Appendix B: Architecture v1](docs/appendix-b-architecture-v1-unscheduled.md)** - The earlier component designs, unscheduled

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
git clone https://github.com/loganathan-sekaran/ebrain.git
cd ebrain
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

See [docs/00-getting-started.md](docs/00-getting-started.md) for the full onboarding checklist.

## 🏗️ Project Status

**Current stage**: Stage 0 (repo hygiene). Horizon 1 starts when it lands.
**Plan**: [docs/PLAN.md](docs/PLAN.md) is the only place that holds months, gates and budgets.

| Horizon | Focus | Gate | Status |
|---|---|---|---|
| H1 | Growth + continual learning (Split-CIFAR-100, class-incremental) | Gate 1 | 📋 Planned |
| H2 | Grounded concepts + agency (MiniGrid/BabyAI) | Gate 2 | 📋 Planned |
| H3 | Deliberation, governed tools, adapter clones | Gate 3 | 📋 Planned |
| Beyond | Expertise, concept transfer, ecosystem | — | Unscheduled |

## 🔬 Research Foundation

E-Brain builds on established AI research areas:

- **Continual Learning**: EWC, A-GEM, Progressive Neural Networks
- **Neural Architecture Search**: DARTS, ENAS, dynamic architectures
- **Meta-Learning**: MAML, learning to learn
- **Multi-Modal Learning**: CLIP, Flamingo, unified embeddings
- **Developmental AI**: Curriculum learning, computational models of cognition
- **Knowledge Distillation**: Teacher-student learning, concept transfer

## 🎓 Use Cases

> **Illustrative long-term use cases only.** None of them are in the H1–H3 plan. Any medical use would need clinical validation, regulatory approval and human oversight.

### Medical AI Network
Clone base model to create specialized medical experts:
- Radiology specialist (X-ray, CT, MRI analysis)
- Pathology specialist (tissue analysis)
- Diagnostic assistant (integrated reasoning)

### Multi-Lingual Tutoring
Create language tutors that share grammatical concepts:
- Spanish tutor teaches "subjunctive mood" concept
- French tutor learns and adapts the concept
- Accelerated learning across language family

### Code Generation Specialists
Domain experts that share programming paradigms:
- Python expert masters async/await
- Transfers pattern to JavaScript expert
- Cross-language knowledge synthesis

## 🤝 Contributing

This is a research project in early planning stages. We welcome:

- **Feedback on architecture**: Review [docs/02-architecture.md](docs/02-architecture.md)
- **Research suggestions**: Relevant papers and techniques
- **Implementation ideas**: Novel approaches to challenges
- **Collaboration**: Researchers and engineers interested in developmental AI

## 📖 Citation

If you use E-Brain in your research, please cite:

```bibtex
@software{ebrain2025,
  title = {E-Brain: Developmental AI with Dynamic Growth},
  author = {Loganathan Sekaran},
  year = {2025},
  url = {https://github.com/loganathan-sekaran/ebrain}
}
```

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgments

Inspired by:
- Cognitive science research on human brain development
- DeepMind's work on continual learning and generalization
- OpenAI's research on large-scale language models
- Meta's Llama and efficient model architectures
- The broader AI research community

## 📞 Contact

- **Project Lead**: Loganathan Sekaran
- **Repository**: [github.com/loganathan-sekaran/ebrain](https://github.com/loganathan-sekaran/ebrain)

---

**Note**: This is an experimental research project. The timeline and scope are ambitious. See [docs/01-feasibility-analysis.md](docs/01-feasibility-analysis.md) for detailed feasibility assessment.

*Last Updated: November 25, 2026*