# Development Strategy: Implementation vs Training

## Overview

E-Brain development follows a clear two-phase approach for each capability:

1. **Implementation Phase**: Build the system components (code infrastructure)
2. **Training Phase**: Train the implemented system with appropriate data

**Critical Principle**: You cannot train an infant brain that doesn't exist yet. The neural architecture, sensory systems, and learning mechanisms must be **implemented first**, then we can begin developmental training.

---

## Development Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: IMPLEMENTATION (No Training Yet)                  │
├─────────────────────────────────────────────────────────────┤
│  1. Code neural architecture (neurons, synapses, STDP)      │
│  2. Code sensory input systems (vision, audio, etc.)        │
│  3. Code learning mechanisms (neurogenesis, pruning)        │
│  4. Code reward system infrastructure                       │
│  5. Write unit tests for each component                     │
│  6. Integration testing (systems work together)             │
│                                                              │
│  OUTPUT: Functional but untrained E-Brain system            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: TRAINING (System Already Exists)                  │
├─────────────────────────────────────────────────────────────┤
│  1. Prepare training data (sensory inputs)                  │
│  2. Design training curriculum (infant → expert)            │
│  3. Run training loops (developmental phases)               │
│  4. Monitor learning progress (metrics, checkpoints)        │
│  5. Evaluate developmental milestones                       │
│  6. Adjust hyperparameters if needed                        │
│                                                              │
│  OUTPUT: Trained E-Brain with learned capabilities          │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Development Strategy

### Stage 1: Foundation (Month 1-3) - IMPLEMENTATION ONLY

#### Month 1-2: Core Infrastructure Implementation

**What We're Coding:**
```python
# Week 1-2: Bio-inspired Neuron Implementation
class BioInspiredNeuron:
    def __init__(self):
        self.dendrites = []  # Input branches
        self.axon = None     # Output
        self.membrane_potential = 0.0
        self.spike_history = []
    
    def add_dendrite(self, source_neuron, weight):
        """Code the dendritic structure"""
        pass
    
    def integrate_inputs(self, inputs, dt=0.001):
        """Code temporal integration (leaky integrator)"""
        pass
    
    def apply_stdp(self, pre_spike_time, post_spike_time):
        """Code STDP learning rule"""
        pass

# Week 3-4: Neurogenesis System Implementation
class NeurogenesisSystem:
    def __init__(self):
        self.neuron_pool = []
        self.growth_signals = {}
    
    def add_neuron(self, layer, position):
        """Code neuron creation"""
        pass
    
    def prune_weak_connections(self, threshold=0.1):
        """Code connection pruning"""
        pass
    
    def hebbian_rewiring(self):
        """Code activity-based rewiring"""
        pass

# Week 5-6: Basic Vision Input System
class VisionInputSystem:
    def __init__(self, resolution=(64, 64)):
        self.resolution = resolution
        self.retina = None  # Will process images
    
    def preprocess_image(self, raw_image):
        """Code image preprocessing"""
        # Resize, normalize, extract features
        pass
    
    def encode_to_spikes(self, image):
        """Code image-to-spike conversion"""
        # Brighter pixels → higher spike rates
        pass

# Week 7-8: Reward System Infrastructure
class RewardSystem:
    def __init__(self):
        self.novelty_detector = NoveltyDetector()
        self.prediction_tracker = PredictionTracker()
    
    def compute_reward(self, state, action, outcome):
        """Code reward computation"""
        pass
```

**No Training Yet!** We're just building the infrastructure.

**Testing Strategy:**
```python
# Unit tests for each component
def test_neuron_spike():
    neuron = BioInspiredNeuron()
    # Test: Does neuron spike when input exceeds threshold?
    assert neuron.check_spike(input=1.5) == True

def test_stdp_strengthens_connection():
    neuron = BioInspiredNeuron()
    # Test: Does STDP strengthen connection when pre→post?
    initial_weight = 0.5
    neuron.apply_stdp(pre_time=100, post_time=120)  # 20ms delay
    assert neuron.weight > initial_weight

def test_vision_encoding():
    vision = VisionInputSystem()
    # Test: Does bright pixel → high spike rate?
    bright_pixel = 255
    spike_rate = vision.encode_to_spikes([[bright_pixel]])
    assert spike_rate > 0.8
```

**Deliverable:** Functional E-Brain codebase with passing unit tests. **No trained model yet.**

---

#### Month 3: Proof of Concept - FIRST TRAINING

**Now We Can Train!** The system exists, so we can feed it data.

**Training Data Needed:**
```python
# Dataset 1: Simple Moving Shapes (Self-Generated)
def generate_training_data():
    """
    Generate 10,000 video frames of simple shapes moving
    """
    data = []
    for i in range(10000):
        # Create frame: black background, white square moving
        frame = np.zeros((64, 64))
        x_pos = (i % 64)
        frame[30:34, x_pos:x_pos+4] = 1.0  # Moving square
        
        # Next frame (prediction target)
        next_frame = np.zeros((64, 64))
        next_x = ((i + 1) % 64)
        next_frame[30:34, next_x:next_x+4] = 1.0
        
        data.append({
            'current_frame': frame,
            'next_frame': next_frame,
            'motion_vector': [1, 0]  # Moving right
        })
    return data

# No external dataset needed - we generate it!
```

**Training Loop:**
```python
# training/phase1_poc.py
def train_phase1_poc():
    """
    First training: Teach E-Brain to predict next frame
    """
    ebrain = EBrainSystem(
        vision_resolution=(64, 64),
        initial_neurons=100,
        enable_neurogenesis=True
    )
    
    training_data = generate_training_data()
    
    for epoch in range(50):
        for sample in training_data:
            # 1. Feed current frame to vision system
            spikes = ebrain.vision.encode_to_spikes(sample['current_frame'])
            
            # 2. E-Brain processes and predicts
            prediction = ebrain.forward(spikes)
            
            # 3. Compare prediction to actual next frame
            target = sample['next_frame']
            prediction_error = mse(prediction, target)
            
            # 4. Compute reward
            reward = -prediction_error  # Good prediction = high reward
            if ebrain.novelty_detector.is_novel(sample['current_frame']):
                reward += 0.5  # Bonus for new patterns
            
            # 5. Apply learning
            ebrain.apply_stdp()  # Strengthen/weaken synapses
            ebrain.update_neuron_growth(reward)  # Grow if learning
            
        # Evaluate every epoch
        if epoch % 10 == 0:
            eval_accuracy = evaluate_prediction(ebrain, test_data)
            print(f"Epoch {epoch}: Accuracy {eval_accuracy:.2f}")
    
    # Save trained model
    ebrain.save_checkpoint("checkpoints/phase1_poc.pt")
```

**Success Criteria:**
- E-Brain predicts next frame position with >70% accuracy
- Neurons grow from 100 → ~500 during training
- Weak connections pruned (connection count decreases initially)
- Novel patterns trigger higher rewards

**Training Time:** 2-4 hours on GPU

**Deliverable:** First trained E-Brain checkpoint that demonstrates basic learning.

---

### Stage 2: MVP Development (Month 4-6) - IMPLEMENTATION THEN TRAINING

#### Month 4: Implement Phase 1 Systems

**Coding Tasks (No Training):**
```python
# 1. Expand Vision System
class AdvancedVisionSystem:
    def __init__(self):
        self.edge_detectors = []
        self.corner_detectors = []
        self.pattern_memory = {}
    
    def detect_edges(self, image):
        """Code edge detection (Gabor filters)"""
        pass
    
    def detect_corners(self, image):
        """Code corner detection (Harris corners)"""
        pass

# 2. Implement Agency Detection
class AgencyDetector:
    def __init__(self):
        self.action_history = []
        self.outcome_history = []
    
    def record_action(self, action, outcome):
        """Track: Did I cause this outcome?"""
        pass
    
    def compute_agency(self):
        """Correlation between my actions and outcomes"""
        pass

# 3. Implement Timing System
class InternalTimingSystem:
    def __init__(self):
        self.millisecond_timer = MillisecondTimer(tick_rate=10)
        self.interval_timer = IntervalTimer()
    
    def record_timestamp(self, event):
        """Record when event occurred"""
        pass

# 4. Implement Sensory Grounding (Basic)
class SensoryGroundingDatabase:
    def __init__(self):
        self.concept_groundings = {}
    
    def store_grounding(self, concept, visual_features):
        """Store: 'ball' → round, red, texture"""
        pass
```

**Testing:**
```python
def test_edge_detection():
    vision = AdvancedVisionSystem()
    # Test edge detector on vertical line
    test_image = create_vertical_line_image()
    edges = vision.detect_edges(test_image)
    assert edges[32, :].sum() > 50  # Strong vertical edge detected

def test_agency_detection():
    agency = AgencyDetector()
    # Test: Detect self-caused outcomes
    agency.record_action(action="move_right", outcome="position_changed")
    agency.record_action(action="stay", outcome="position_changed")  # External
    
    assert agency.compute_agency()["move_right"] > 0.8  # High agency
    assert agency.compute_agency()["stay"] < 0.2  # Low agency
```

**Deliverable:** Extended E-Brain system with Phase 1 capabilities coded. **Not trained yet.**

---

#### Month 5-6: Phase 1 Training (Infant Stage)

**Training Data Needed:**

```python
# Dataset 1: MNIST (Handwritten Digits) - For Vision
# Source: http://yann.lecun.com/exdb/mnist/
# Size: 60,000 training images, 10,000 test images
# Format: 28x28 grayscale images

def prepare_mnist_data():
    """
    Prepare MNIST for E-Brain training
    """
    from torchvision import datasets, transforms
    
    mnist_train = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to E-Brain resolution
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    return mnist_train

# Dataset 2: BabyAI (Simple Grid Navigation) - For Agency
# Source: https://github.com/mila-iqia/babyai
# Task: Navigate grid, pick up objects, follow simple commands

def setup_babyai_environment():
    """
    Setup BabyAI for agency learning
    """
    import gym
    import gym_minigrid
    
    env = gym.make('BabyAI-GoToObj-v0')
    # E-Brain learns: "I moved right" → "Position changed"
    return env

# Dataset 3: Simple Audio (For Multimodal) - OPTIONAL for Month 6
# Source: Speech Commands Dataset
# Size: 65,000 one-second audio clips (yes, no, up, down, etc.)
```

**Training Curriculum:**

```python
# training/phase1_infant.py

def train_phase1_infant():
    """
    Train E-Brain through infant stage (Phase 1)
    """
    ebrain = EBrainSystem.load_checkpoint("checkpoints/phase1_poc.pt")
    
    # Curriculum Part 1: Vision Learning (Week 1-2)
    mnist_data = prepare_mnist_data()
    
    print("WEEK 1-2: Learning to recognize digits...")
    for epoch in range(20):
        for image, label in mnist_data:
            # 1. Process image
            spikes = ebrain.vision.encode_to_spikes(image)
            
            # 2. E-Brain predicts: "What is this?"
            prediction = ebrain.forward(spikes)
            
            # 3. Reward based on recognition
            correct = (prediction.argmax() == label)
            reward = 1.0 if correct else 0.0
            
            # 4. Novelty bonus
            if ebrain.novelty_detector.is_novel_digit(image):
                reward += 0.5
            
            # 5. Apply learning
            ebrain.learn(reward)
        
        # Evaluate edge/corner detection
        eval_edges = evaluate_edge_detection(ebrain)
        eval_corners = evaluate_corner_detection(ebrain)
        print(f"Epoch {epoch}: Edges={eval_edges:.2f}, Corners={eval_corners:.2f}")
    
    # Curriculum Part 2: Agency Learning (Week 3-4)
    babyai_env = setup_babyai_environment()
    
    print("WEEK 3-4: Learning 'I caused this'...")
    for episode in range(1000):
        state = babyai_env.reset()
        done = False
        
        while not done:
            # 1. E-Brain chooses action
            action = ebrain.decide_action(state)
            
            # 2. Execute in environment
            next_state, reward_ext, done, info = babyai_env.step(action)
            
            # 3. Agency detection
            agency_score = ebrain.agency_detector.compute_agency()
            
            # 4. Intrinsic reward for agency discovery
            reward = reward_ext + 0.5 * agency_score
            
            # 5. Learn
            ebrain.learn(reward)
            state = next_state
    
    # Curriculum Part 3: Sensory Grounding (Week 5-6)
    print("WEEK 5-6: Grounding concepts to sensory features...")
    
    for concept, examples in concept_examples.items():
        # concept = "circle", examples = [circle images]
        for image in examples:
            # Extract visual features
            features = ebrain.vision.extract_features(image)
            
            # Ground concept
            ebrain.sensory_grounding.store_grounding(
                concept=concept,
                visual_features=features
            )
    
    # Phase 1 Complete!
    ebrain.save_checkpoint("checkpoints/phase1_complete.pt")
    
    print("Phase 1 Training Complete!")
    print(f"- Neurons: {ebrain.neuron_count}")
    print(f"- Concepts grounded: {len(ebrain.sensory_grounding.concepts)}")
    print(f"- MNIST accuracy: {evaluate_mnist(ebrain):.2%}")
    print(f"- BabyAI success rate: {evaluate_babyai(ebrain):.2%}")
```

**Data Requirements Summary:**
- **MNIST**: 60K images (included, no cost)
- **BabyAI**: Synthetic environment (generated on-the-fly, no storage)
- **Sensory grounding examples**: 100-500 curated images per concept (self-collected or ImageNet subset)

**Training Time:** 
- Vision: 10-20 hours (20 epochs × 60K images)
- Agency: 5-10 hours (1000 episodes × ~100 steps)
- Grounding: 1-2 hours (small dataset)
- **Total: ~30-40 hours of training**

**Hardware:** 1x GPU (RTX 3090 or better recommended)

**Success Criteria:**
- ✅ MNIST recognition >85% (edges, corners working)
- ✅ BabyAI navigation >70% success rate
- ✅ Agency detection >80% accuracy ("I caused this" vs external)
- ✅ 50+ concepts grounded (circle, square, red, blue, etc.)

---

### Stage 3: Multi-Modal Integration (Month 7-10)

#### Month 7: Implement Language Systems

**Coding Tasks:**
```python
# 1. Language Encoder
class LanguageEncoder:
    def __init__(self, vocab_size=10000):
        self.vocabulary = {}
        self.embeddings = None
    
    def tokenize(self, text):
        """Convert text to tokens"""
        pass
    
    def encode_to_spikes(self, tokens):
        """Convert tokens to spike patterns"""
        pass

# 2. Concept Hierarchy (Level 2)
class ConceptHierarchy:
    def __init__(self):
        self.level1_concepts = {}  # Parts
        self.level2_concepts = {}  # Objects
        self.relationships = {}
    
    def learn_object_from_parts(self, object_name, parts):
        """'car' = wheels + body + windows"""
        pass

# 3. Theory of Mind System
class TheoryOfMindSystem:
    def __init__(self):
        self.entity_beliefs = {}  # What does X know?
        self.entity_goals = {}    # What does X want?
    
    def track_belief(self, entity, belief):
        """Track what another entity knows"""
        pass
    
    def infer_goal(self, entity, actions):
        """Infer goal from observed actions"""
        pass
```

**No training yet** - just implementing the infrastructure.

---

#### Month 8-10: Phase 2-3 Training (Child Language Acquisition)

**Training Data Needed:**

```python
# Dataset 1: WikiText-103 (Language Corpus)
# Source: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
# Size: 100M tokens
# Format: Plain text articles

def prepare_wikitext_data():
    """
    Prepare language data for E-Brain
    """
    from datasets import load_dataset
    
    wikitext = load_dataset('wikitext', 'wikitext-103-v1')
    
    # Filter for simple sentences (child-appropriate)
    simple_sentences = []
    for text in wikitext['train']['text']:
        # Keep sentences with common words only
        if is_simple_sentence(text):
            simple_sentences.append(text)
    
    return simple_sentences[:100000]  # 100K simple sentences

# Dataset 2: COCO Captions (Vision-Language Grounding)
# Source: https://cocodataset.org/
# Size: 120K images with 5 captions each
# Format: Image + text description

def prepare_coco_data():
    """
    Link images to language descriptions
    """
    from pycocotools.coco import COCO
    
    coco = COCO('annotations/captions_train2017.json')
    
    vision_language_pairs = []
    for img_id in coco.getImgIds():
        # Get image
        img = coco.loadImgs(img_id)[0]
        
        # Get captions
        ann_ids = coco.getAnnIds(imgIds=img_id)
        captions = [ann['caption'] for ann in coco.loadAnns(ann_ids)]
        
        vision_language_pairs.append({
            'image': img,
            'captions': captions
        })
    
    return vision_language_pairs

# Dataset 3: Sally-Anne Test (Theory of Mind)
# Source: Hand-crafted scenarios
# Size: 100 scenarios

def create_theory_of_mind_scenarios():
    """
    Create scenarios to test Theory of Mind
    """
    scenarios = [
        {
            'setup': "Sally puts ball in basket. Sally leaves. Anne moves ball to box.",
            'question': "Where will Sally look for the ball?",
            'correct_answer': "basket",  # Sally's belief, not reality
            'theory_of_mind_required': True
        },
        # ... 99 more scenarios
    ]
    return scenarios
```

**Training Curriculum:**

```python
# training/phase2_phase3.py

def train_phase2_phase3():
    """
    Train language and social cognition
    """
    ebrain = EBrainSystem.load_checkpoint("checkpoints/phase1_complete.pt")
    
    # Enable Phase 2 systems
    ebrain.enable_language_processing()
    ebrain.enable_concurrent_thoughts(num_thoughts=2)
    
    # PHASE 2: Basic Language (Month 8)
    print("PHASE 2: Learning language...")
    
    simple_sentences = prepare_wikitext_data()
    
    for epoch in range(30):
        for sentence in simple_sentences:
            # 1. Process sentence
            tokens = ebrain.language.tokenize(sentence)
            spikes = ebrain.language.encode_to_spikes(tokens)
            
            # 2. Predict next word
            prediction = ebrain.forward(spikes)
            
            # 3. Reward for correct prediction
            target = tokens[1:]  # Next word sequence
            accuracy = compute_accuracy(prediction, target)
            reward = accuracy
            
            # 4. Learn
            ebrain.learn(reward)
        
        vocab_size = len(ebrain.language.vocabulary)
        perplexity = evaluate_language_model(ebrain)
        print(f"Epoch {epoch}: Vocab={vocab_size}, Perplexity={perplexity:.2f}")
    
    # PHASE 2.5: Vision-Language Grounding (Month 9)
    print("PHASE 2.5: Linking vision to language...")
    
    coco_data = prepare_coco_data()
    
    for epoch in range(20):
        for sample in coco_data:
            # 1. Process image
            visual_spikes = ebrain.vision.encode(sample['image'])
            
            # 2. Process caption
            caption_spikes = ebrain.language.encode(sample['captions'][0])
            
            # 3. Concurrent thoughts!
            thought1 = ebrain.concurrent_thoughts.create_thought(
                task="understand_image",
                input=visual_spikes
            )
            thought2 = ebrain.concurrent_thoughts.create_thought(
                task="understand_caption",
                input=caption_spikes
            )
            
            # 4. Bind visual concepts to words
            visual_concepts = thought1.extract_concepts()
            word_concepts = thought2.extract_concepts()
            
            ebrain.concept_hierarchy.link_concepts(visual_concepts, word_concepts)
            
            # 5. Reward for successful grounding
            grounding_accuracy = evaluate_grounding(ebrain, sample)
            ebrain.learn(grounding_accuracy)
    
    # PHASE 3: Theory of Mind (Month 10)
    print("PHASE 3: Learning Theory of Mind...")
    
    ebrain.enable_theory_of_mind()
    ebrain.expand_concurrent_thoughts(num_thoughts=4)
    
    tom_scenarios = create_theory_of_mind_scenarios()
    
    for epoch in range(50):
        for scenario in tom_scenarios:
            # 1. Process scenario
            setup_spikes = ebrain.language.encode(scenario['setup'])
            question_spikes = ebrain.language.encode(scenario['question'])
            
            # 2. Track beliefs
            ebrain.theory_of_mind.process_scenario(setup_spikes)
            # Now E-Brain tracks: Sally believes ball is in basket
            
            # 3. Answer question
            answer = ebrain.answer_question(question_spikes)
            
            # 4. Reward for correct Theory of Mind reasoning
            correct = (answer == scenario['correct_answer'])
            reward = 2.0 if correct else 0.0
            
            ebrain.learn(reward)
    
    ebrain.save_checkpoint("checkpoints/phase3_complete.pt")
    
    print("Phase 2-3 Complete!")
    print(f"- Vocabulary: {len(ebrain.language.vocabulary)} words")
    print(f"- Concurrent thoughts: {ebrain.concurrent_thoughts.capacity}")
    print(f"- Theory of Mind accuracy: {evaluate_tom(ebrain):.2%}")
```

**Data Requirements:**
- **WikiText-103**: 100M tokens (~500MB download)
- **COCO Captions**: 120K images (~20GB download)
- **Theory of Mind scenarios**: Self-generated (no download)

**Training Time:**
- Language: 40-60 hours (30 epochs × 100K sentences)
- Vision-Language: 30-50 hours (20 epochs × 120K pairs)
- Theory of Mind: 10-15 hours (50 epochs × 100 scenarios)
- **Total: ~100-150 hours**

**Hardware:** 2x GPU recommended for concurrent processing

---

### Stage 4-5: Abstract Reasoning & Expertise (Month 11-20)

#### Implementation Strategy

**Month 11-12: Code Phase 4 Systems**
```python
# Implement without training:
# - Abstract reasoning modules
# - Circadian clock system
# - 7-slot concurrent thought system
# - Rich multimodal sensory grounding
# - E-Brain-to-E-Brain communication
```

**Month 13-20: Training Phase 4-5**

**Training Data Needed:**

```python
# Dataset 1: RAVEN's Progressive Matrices (Abstract Reasoning)
# Source: https://github.com/WellyZhang/RAVEN
# Size: 70K matrix problems
# Task: Pattern completion (visual abstract reasoning)

# Dataset 2: GSM8K (Math Word Problems)
# Source: https://github.com/openai/grade-school-math
# Size: 8K problems
# Task: Multi-step math reasoning

# Dataset 3: HotpotQA (Multi-Hop Reasoning)
# Source: https://hotpotqa.github.io/
# Size: 113K question-answer pairs
# Task: Reasoning across multiple documents

# Dataset 4: Expert Domain Data (Specialization)
# Examples:
# - arXiv papers (for research expertise)
# - Stack Overflow (for programming expertise)
# - Medical journals (for medical expertise)
# - Legal documents (for legal expertise)

def train_phase4_phase5():
    """
    Train abstract reasoning and expertise
    """
    ebrain = EBrainSystem.load_checkpoint("checkpoints/phase3_complete.pt")
    
    # Enable Phase 4 systems
    ebrain.expand_concurrent_thoughts(num_thoughts=7)
    ebrain.enable_circadian_clock()
    ebrain.enable_rich_sensory_grounding()
    
    # Train on abstract reasoning
    ravens = load_ravens_dataset()
    gsm8k = load_gsm8k_dataset()
    hotpotqa = load_hotpotqa_dataset()
    
    # Multi-month training loop
    for month in range(8):  # Month 13-20
        # Circadian cycle: Active learning during "day"
        for day in range(30):
            # Active period: 6am-10pm (16 hours)
            for hour in range(16):
                # Train on various tasks
                train_on_ravens(ebrain, ravens, hours=1)
                train_on_gsm8k(ebrain, gsm8k, hours=1)
                train_on_hotpotqa(ebrain, hotpotqa, hours=1)
            
            # Sleep period: 10pm-6am (8 hours)
            ebrain.enter_sleep_cycle(duration_hours=8)
            # Consolidation happens during sleep
        
        # Specialization in chosen domain (Month 17+)
        if month >= 4:
            domain_data = load_domain_data(ebrain.chosen_domain)
            train_on_domain(ebrain, domain_data, hours=100)
    
    ebrain.save_checkpoint("checkpoints/phase5_expert.pt")
```

**Training Time:**
- Phase 4: ~300 hours (abstract reasoning, 3 months)
- Phase 5: ~500 hours (expertise, 5 months)
- **Total: ~800 hours** (~33 days of continuous training)

---

## Data Requirements Summary

### Datasets Overview

| Dataset | Size | Purpose | Phase | Cost |
|---------|------|---------|-------|------|
| **Generated Shapes** | 10K frames | Proof of concept | POC | Free (generated) |
| **MNIST** | 60K images | Vision learning | Phase 1 | Free |
| **BabyAI** | Synthetic | Agency learning | Phase 1 | Free (generated) |
| **Concept Examples** | 10K images | Sensory grounding | Phase 1 | Free (curated) |
| **WikiText-103** | 100M tokens | Language learning | Phase 2 | Free |
| **COCO Captions** | 120K images | Vision-language | Phase 2-3 | Free |
| **Theory of Mind** | 100 scenarios | Social cognition | Phase 3 | Free (hand-crafted) |
| **RAVEN** | 70K problems | Abstract reasoning | Phase 4 | Free |
| **GSM8K** | 8K problems | Math reasoning | Phase 4 | Free |
| **HotpotQA** | 113K pairs | Multi-hop reasoning | Phase 4 | Free |
| **Domain Data** | Varies | Expertise | Phase 5 | Free (public archives) |

**Total Cost: $0** - All datasets are publicly available!

**Total Storage: ~50GB**

---

## Development Team Structure

### Recommended Team Roles

```
┌─────────────────────────────────────────────────────────┐
│  Role 1: Architecture Engineer                          │
│  - Implements neural systems (neurons, STDP, growth)    │
│  - Codes sensory systems (vision, language, etc.)       │
│  - Builds core infrastructure                           │
│  Skills: PyTorch, neuroscience, systems programming     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Role 2: Learning Systems Engineer                      │
│  - Implements reward systems                            │
│  - Codes learning algorithms (STDP, RL, etc.)           │
│  - Designs training curricula                           │
│  Skills: ML, reinforcement learning, optimization       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Role 3: Data & Training Engineer                       │
│  - Prepares datasets                                    │
│  - Runs training loops                                  │
│  - Monitors training progress                           │
│  Skills: Data engineering, MLOps, distributed training  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Role 4: Evaluation Engineer                            │
│  - Designs evaluation metrics                           │
│  - Runs developmental tests                             │
│  - Validates milestones                                 │
│  Skills: Testing, benchmarking, data analysis           │
└─────────────────────────────────────────────────────────┘
```

**Minimum Team Size: 2-3 engineers** (roles can be combined)

---

## Timeline Summary

```
Month 1-2:  IMPLEMENT core infrastructure           ⚙️ CODE
Month 3:    TRAIN proof of concept                  🎓 TRAIN (2-4 hours)
Month 4:    IMPLEMENT Phase 1 systems               ⚙️ CODE
Month 5-6:  TRAIN Phase 1 (infant)                  🎓 TRAIN (~40 hours)
Month 7:    IMPLEMENT language systems              ⚙️ CODE
Month 8-10: TRAIN Phase 2-3 (child language)        🎓 TRAIN (~150 hours)
Month 11-12: IMPLEMENT Phase 4 systems              ⚙️ CODE
Month 13-16: TRAIN Phase 4 (abstract reasoning)     🎓 TRAIN (~300 hours)
Month 17-20: TRAIN Phase 5 (expertise)              🎓 TRAIN (~500 hours)

Total Development Time: 20 months
Total Coding Time: 6 months
Total Training Time: 14 months (~1000 GPU hours)
```

---

## Key Principles

### 1. Implementation Before Training
**You cannot train a system that doesn't exist.** Always code the infrastructure first, test it with unit tests, then begin training.

### 2. Synthetic Data First
Start with self-generated data (moving shapes, BabyAI environments) before using real-world datasets. This lets you validate the system without downloading large datasets.

### 3. Incremental Complexity
Don't train Phase 5 capabilities on day 1. Follow the developmental progression:
- Phase 1: Infant (simple vision, agency)
- Phase 2: Child (language basics)
- Phase 3: Child (social cognition)
- Phase 4: Adult (abstract reasoning)
- Phase 5: Expert (specialization)

### 4. Checkpoint Everything
Save checkpoints after every phase. If training fails, you can resume from the last successful phase.

### 5. Evaluation Metrics
Define success criteria **before** training:
- Phase 1: >85% MNIST accuracy
- Phase 2: >1000 word vocabulary
- Phase 3: >80% Theory of Mind accuracy
- Phase 4: >70% RAVEN accuracy
- Phase 5: Domain-specific benchmarks

---

## Next Steps

1. **Month 1**: Start coding `BioInspiredNeuron` class
2. **Month 1**: Write unit tests for neuron behavior
3. **Month 2**: Implement `NeurogenesisSystem` class
4. **Month 2**: Implement `VisionInputSystem` class
5. **Month 3**: Generate training data and run first training loop
6. **Month 3**: Evaluate POC and iterate

**The journey begins with code, not training data!** 🚀
