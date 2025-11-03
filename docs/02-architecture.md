# E-Brain Technical Architecture

## Overview

E-Brain is designed as a modular, dynamically growing neural system that mimics developmental stages of human learning. The architecture emphasizes flexibility, growth capability, and multi-modal integration.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        E-Brain System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────── INPUT LAYER ──────────────────┐          │
│  │                                                    │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │          │
│  │  │   Vision    │  │    Audio    │  │   Text   │ │          │
│  │  │   Encoder   │  │   Encoder   │  │  Encoder │ │          │
│  │  └──────┬──────┘  └──────┬──────┘  └────┬─────┘ │          │
│  └─────────┼─────────────────┼──────────────┼───────┘          │
│            │                 │              │                    │
│            └─────────────────┴──────────────┘                    │
│                             │                                     │
│  ┌───────────────────── CENTRAL EXECUTIVE ─────────────────────┐│
│  │                                                               ││
│  │  ┌──────────────────────────────────────────────────────┐  ││
│  │  │           Unified Embedding Space                     │  ││
│  │  └────────────────────┬─────────────────────────────────┘  ││
│  │                       │                                      ││
│  │  ┌────────────────────┴─────────────────────────────────┐  ││
│  │  │          Working Memory (Attention Layer)            │  ││
│  │  └────────────────────┬─────────────────────────────────┘  ││
│  │                       │                                      ││
│  │  ┌────────────────────┴─────────────────────────────────┐  ││
│  │  │        Dynamic Neural Core (Growable Layers)         │  ││
│  │  │    ┌──────────┐  ┌──────────┐  ┌──────────┐         │  ││
│  │  │    │ Module 1 │  │ Module 2 │  │ Module N │  [+]    │  ││
│  │  │    └──────────┘  └──────────┘  └──────────┘         │  ││
│  │  │                                                       │  ││
│  │  │    ┌───────────────────────────────────────────┐    │  ││
│  │  │    │   Multi-Stage Reasoning System            │    │  ││
│  │  │    │   [Stage 0→1→2→3→4] (Adaptive Depth)     │    │  ││
│  │  │    └───────────────────────────────────────────┘    │  ││
│  │  └────────────────────┬─────────────────────────────────┘  ││
│  │                       │                                      ││
│  │  ┌────────────────────┴─────────────────────────────────┐  ││
│  │  │         Long-Term Memory (Vector Database)           │  ││
│  │  │  - Episodic Memory  - Semantic Memory - Skills       │  ││
│  │  └──────────────────────────────────────────────────────┘  ││
│  │                                                               ││
│  └───────────────────────────────────┬───────────────────────────┘│
│                                      │                            │
│  ┌────────────────── META-LEARNING SYSTEM ─────────────────────┐ │
│  │                                                               │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                │ │
│  │  │  Performance     │  │  Growth          │                │ │
│  │  │  Monitor         │  │  Controller      │                │ │
│  │  └────────┬─────────┘  └────────┬─────────┘                │ │
│  │           │                     │                           │ │
│  │  ┌────────┴─────────┐  ┌────────┴─────────┐                │ │
│  │  │  Uncertainty     │  │  Pruning         │                │ │
│  │  │  Estimator       │  │  System          │                │ │
│  │  └──────────────────┘  └──────────────────┘                │ │
│  │                                                               │ │
│  └───────────────────────────────────┬───────────────────────────┘ │
│                                      │                            │
│  ┌─────────────────── OUTPUT LAYER ──────────────┴────────────┐ │
│  │                                                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐        │ │
│  │  │  Language   │  │  Decision   │  │    Motor     │        │ │
│  │  │  Generator  │  │   Making    │  │   Control    │        │ │
│  │  └─────────────┘  └─────────────┘  └──────────────┘        │ │
│  │                                                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────── CORRECTION MECHANISM ─────────────────────┐  │
│  │  Error Detection → Feedback Integration → Self-Correction │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Input Layer - Sensory Encoders

#### Vision Encoder
- **Architecture:** CNN + Vision Transformer hybrid
- **Capabilities:** 
  - Raw image/video processing
  - Feature extraction at multiple scales
  - Object detection and segmentation
- **Growth Strategy:** Add specialized layers for new visual tasks

```python
class VisionEncoder:
    - Convolutional backbone (ResNet, EfficientNet)
    - Vision Transformer (ViT) for global context
    - Feature pyramid for multi-scale processing
    - Dynamic layer addition for new visual concepts
```

#### Audio Encoder
- **Architecture:** Spectrogram CNN + Transformer
- **Capabilities:**
  - Speech recognition
  - Sound event detection
  - Music understanding
- **Growth Strategy:** Task-specific heads for specialized audio tasks

```python
class AudioEncoder:
    - Mel-spectrogram converter
    - Temporal CNN for local patterns
    - Transformer for long-range dependencies
    - Language-specific phoneme detectors
```

#### Text Encoder
- **Architecture:** Transformer-based (BERT-style)
- **Capabilities:**
  - Token embedding
  - Contextual understanding
  - Multi-lingual support (when taught)
- **Growth Strategy:** Expand vocabulary and language-specific layers

```python
class TextEncoder:
    - Tokenizer (BPE or WordPiece)
    - Embedding layer (expandable)
    - Transformer encoder stack
    - Language detection module
```

### 2. Central Executive - The Brain Core

#### Unified Embedding Space
- **Purpose:** Project all modalities into common representation space
- **Architecture:** Cross-modal alignment network
- **Key Feature:** Enables cross-modal reasoning and transfer

```python
class UnifiedEmbedding:
    - Modality-specific projection heads
    - Contrastive learning for alignment
    - Shared latent space (512-2048 dimensions)
    - Cross-modal attention mechanisms
```

#### Working Memory (Attention Layer)
- **Purpose:** Maintain context and focus attention
- **Architecture:** Multi-head attention with memory bank
- **Capacity:** Expandable based on task complexity

```python
class WorkingMemory:
    - Short-term buffer (limited capacity)
    - Multi-head self-attention
    - Cross-attention to long-term memory
    - Attention weights as importance signals
```

#### Dynamic Neural Core (Growable Layers)
- **Purpose:** Main reasoning and processing unit
- **Architecture:** Modular transformer/MLP blocks
- **Growth Mechanism:** Progressive layer addition

```python
class DynamicCore:
    - Base modules (minimum capacity)
    - Growth triggers:
      * Performance plateau
      * High uncertainty
      * Task complexity increase
    - Module types:
      * General-purpose transformers
      * Task-specific experts
      * Cross-modal fusion layers
    - Pruning strategy:
      * Remove low-importance connections
      * Consolidate redundant modules
```

**Growth Algorithm:**
```
1. Monitor performance on current task
2. If performance < threshold AND gradient plateau:
   a. Calculate capacity utilization
   b. If utilization > 80%:
      - Add new module/layer
      - Initialize with smart defaults
3. Else if performance sufficient:
   - Continue training existing modules
4. Periodically prune low-importance weights
```

#### Multi-Stage Reasoning System
- **Purpose:** Enable iterative, deliberate thinking for complex concept understanding and problem solving
- **Architecture:** Iterative refinement loop with depth control
- **Activation:** Triggered by task complexity or uncertainty threshold

```python
class MultiStageReasoning:
    """
    Implements iterative multi-pass reasoning for:
    - Deep concept understanding
    - Complex problem decomposition
    - Chain-of-thought reasoning
    - Self-verification and refinement
    """
    
    def __init__(self, max_stages=5, uncertainty_threshold=0.3):
        self.max_stages = max_stages
        self.uncertainty_threshold = uncertainty_threshold
        self.stage_processors = nn.ModuleList([
            StageProcessor(depth=i) for i in range(max_stages)
        ])
        self.depth_controller = DepthController()
        self.verification_module = VerificationModule()
        
    def forward(self, input_state, task_type):
        """
        Multi-stage reasoning pipeline:
        1. Initial shallow pass
        2. Iterative deepening based on confidence
        3. Verification and refinement
        4. Return best reasoning trace
        """
        reasoning_trace = []
        current_state = input_state
        
        for stage in range(self.max_stages):
            # Stage processing
            stage_output = self.stage_processors[stage](
                current_state,
                context=reasoning_trace
            )
            
            # Add to reasoning trace
            reasoning_trace.append({
                'stage': stage,
                'output': stage_output.prediction,
                'confidence': stage_output.confidence,
                'rationale': stage_output.intermediate_steps
            })
            
            # Check if we can stop early
            if self.should_stop(stage_output, stage):
                break
                
            # Prepare for next stage
            current_state = self.refine_state(
                current_state, 
                stage_output
            )
        
        # Final verification
        verified_output = self.verification_module(
            reasoning_trace,
            input_state
        )
        
        return verified_output, reasoning_trace


class StageProcessor(nn.Module):
    """Individual stage of multi-stage reasoning"""
    
    STAGE_FOCUS = {
        0: "surface_features",      # Quick pattern matching
        1: "relationships",          # Connect concepts
        2: "deep_analysis",          # Abstract reasoning
        3: "integration",            # Synthesize with knowledge
        4: "verification"            # Self-check and refine
    }
    
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.focus = self.STAGE_FOCUS[depth]
        self.processor = self._build_processor()
        
    def forward(self, state, context):
        """Process current stage with access to previous stages"""
        # Focus on stage-specific aspects
        focused_features = self.apply_stage_focus(state)
        
        # Incorporate previous reasoning
        if context:
            contextualized = self.integrate_context(
                focused_features, 
                context
            )
        else:
            contextualized = focused_features
            
        # Generate stage output
        output = self.processor(contextualized)
        
        return StageOutput(
            prediction=output.prediction,
            confidence=output.confidence,
            intermediate_steps=output.chain_of_thought,
            attention_weights=output.attention
        )


class DepthController(nn.Module):
    """Decides how many stages needed based on task complexity"""
    
    def __init__(self):
        super().__init__()
        self.complexity_estimator = ComplexityEstimator()
        
    def determine_depth(self, task, current_confidence):
        """
        Decision logic:
        - Simple tasks: 1-2 stages
        - Medium complexity: 2-3 stages  
        - Complex reasoning: 3-5 stages
        - Low confidence: Force deeper reasoning
        """
        complexity = self.complexity_estimator(task)
        
        if current_confidence > 0.9 and complexity < 0.3:
            return 1  # Fast path for simple tasks
        elif current_confidence > 0.7 and complexity < 0.6:
            return 2  # Medium path
        else:
            return min(5, int(3 + complexity * 2))


class VerificationModule(nn.Module):
    """Verify reasoning consistency and quality"""
    
    def forward(self, reasoning_trace, original_input):
        """
        Verification strategies:
        1. Consistency check across stages
        2. Logical coherence validation
        3. Confidence calibration
        4. Self-contradiction detection
        """
        # Check for contradictions
        contradictions = self.detect_contradictions(reasoning_trace)
        
        # Validate logical flow
        coherence_score = self.check_coherence(reasoning_trace)
        
        # Select best reasoning path
        if contradictions or coherence_score < 0.5:
            # Trigger refinement
            refined = self.refine_reasoning(
                reasoning_trace, 
                original_input
            )
            return refined
        else:
            # Return final stage output
            return reasoning_trace[-1]
```

**Multi-Stage Reasoning Modes:**

1. **Concept Learning Mode** (Progressive Depth)
   ```
   Stage 0: Visual/Surface patterns → "This is round"
   Stage 1: Attribute extraction → "Round, red, has stem"
   Stage 2: Category formation → "Similar to other fruits"
   Stage 3: Conceptual integration → "Apple - fruit category"
   Stage 4: Knowledge consolidation → Store in semantic memory
   ```

2. **Problem Solving Mode** (Decomposition)
   ```
   Stage 0: Problem understanding → Identify goal
   Stage 1: Decomposition → Break into subproblems
   Stage 2: Solution generation → Solve each subproblem
   Stage 3: Integration → Combine solutions
   Stage 4: Verification → Test and validate
   ```

3. **Chain-of-Thought Mode** (Step-by-Step)
   ```
   Stage 0: Initial analysis → "Given: A train leaves at 2pm"
   Stage 1: Extract facts → "Speed: 60mph, Distance: 180mi"
   Stage 2: Apply knowledge → "Time = Distance / Speed"
   Stage 3: Calculate → "180 / 60 = 3 hours"
   Stage 4: Conclude → "Arrives at 5pm"
   ```

4. **Uncertainty Resolution Mode** (Iterative Refinement)
   ```
   Stage 0: Low confidence (0.4) → Identify knowledge gaps
   Stage 1: Retrieve relevant knowledge → Query long-term memory
   Stage 2: Re-analyze with context → Confidence increases (0.6)
   Stage 3: Deep reasoning → More connections found
   Stage 4: High confidence (0.85) → Return result
   ```

**Integration with Existing Components:**

```python
class DynamicCore:
    """Enhanced with multi-stage reasoning"""
    
    def __init__(self):
        super().__init__()
        self.base_modules = nn.ModuleList([...])
        self.multi_stage_reasoner = MultiStageReasoning(
            max_stages=5,
            uncertainty_threshold=0.3
        )
        self.task_complexity_detector = TaskComplexityDetector()
        
    def forward(self, x, task_metadata):
        # Quick pass through base modules
        base_output = self.process_base(x)
        
        # Determine if multi-stage reasoning needed
        if self.needs_deep_reasoning(base_output, task_metadata):
            # Engage multi-stage reasoning
            final_output, reasoning_trace = self.multi_stage_reasoner(
                input_state=base_output,
                task_type=task_metadata['type']
            )
            
            # Log reasoning trace for interpretability
            self.log_reasoning(reasoning_trace)
            
            return final_output
        else:
            # Fast path for simple tasks
            return base_output
            
    def needs_deep_reasoning(self, output, metadata):
        """Decide when to engage multi-stage reasoning"""
        return (
            output.confidence < 0.7 or  # Low confidence
            metadata['complexity'] > 0.5 or  # Complex task
            metadata['requires_reasoning'] or  # Explicit requirement
            output.uncertainty > self.uncertainty_threshold
        )
```

**Benefits:**
- **Interpretability:** Reasoning trace shows how E-Brain thinks
- **Accuracy:** Iterative refinement improves quality
- **Adaptability:** Depth adjusts to task complexity
- **Efficiency:** Fast path for simple tasks, deep reasoning when needed
- **Human-like:** Mimics human deliberate thinking process

#### Long-Term Memory
- **Purpose:** Store learned knowledge persistently
- **Architecture:** Vector database + episodic buffer + concept hierarchy graph

```python
class LongTermMemory:
    # Semantic Memory
    - Vector database (FAISS, Pinecone)
    - Embedding store for facts and concepts
    - Hierarchical organization
    - **Concept Hierarchy Graph (see Concept-Driven Learning below)**
    
    # Episodic Memory
    - Experience replay buffer
    - Important moments storage
    - Temporal context preservation
    
    # Procedural Memory (Skills)
    - Learned policies for tasks
    - Action sequences
    - Strategy templates
```

### 3. Meta-Learning System

#### Performance Monitor
- **Purpose:** Track learning progress and effectiveness
- **Metrics:**
  - Task accuracy/loss
  - Learning rate
  - Generalization gap
  - Forgetting metrics

```python
class PerformanceMonitor:
    - Task-wise metrics tracking
    - Learning curve analysis
    - Forgetting detection
    - Benchmark comparison
```

#### Growth Controller
- **Purpose:** Decide when and how to grow the network
- **Strategies:**
  - Uncertainty-based growth
  - Performance-based growth
  - Complexity-based growth

```python
class GrowthController:
    def should_grow():
        - Check uncertainty levels
        - Check capacity utilization
        - Check performance plateau
        return decision
    
    def grow(module_type):
        - Add new module
        - Initialize weights
        - Connect to existing modules
        - Update architecture registry
```

#### Uncertainty Estimator
- **Purpose:** Know what the model doesn't know
- **Techniques:**
  - Ensemble disagreement
  - Bayesian approximations
  - Calibration metrics

```python
class UncertaintyEstimator:
    - Monte Carlo dropout
    - Ensemble predictions
    - Confidence calibration
    - Out-of-distribution detection
```

### 3.5 Concept-Driven Learning System

**Core Philosophy:** Learning progresses from simple atomic concepts to complex hierarchical concepts through composition and correlation, mimicking human conceptual development.

#### Hierarchical Concept Graph
- **Purpose:** Organize knowledge as a graph of concepts with parent-child relationships
- **Architecture:** Directed acyclic graph (DAG) with concept nodes and relationship edges

```python
class ConceptNode:
    """
    Represents a single concept at any level of abstraction
    """
    def __init__(self, concept_id, name, level):
        self.id = concept_id
        self.name = name
        self.level = level  # 0=atomic, 1=basic, 2=intermediate, 3=abstract, 4=expert
        
        # Concept representation
        self.embedding = None  # Dense vector (512-1024 dim)
        self.prototype = None  # Prototypical example
        self.attributes = {}   # Key properties
        
        # Hierarchical relationships
        self.parent_concepts = []  # What this is part of
        self.child_concepts = []   # What this contains
        self.sibling_concepts = []  # Related at same level
        
        # Compositional rules
        self.composition_rule = None  # How children combine to form this
        self.correlation_weights = {}  # Strength of child relationships
        
        # Learning metadata
        self.confidence = 0.0  # How well learned (0-1)
        self.examples_seen = 0
        self.last_activated = None
        self.activation_count = 0


class ConceptHierarchy:
    """
    Manages the entire concept graph and learning progression
    """
    def __init__(self):
        self.concepts = {}  # concept_id -> ConceptNode
        self.graph = nx.DiGraph()  # NetworkX graph for traversal
        self.concept_levels = {
            0: [],  # Atomic: color, shape, edge, sound
            1: [],  # Basic: object parts, phonemes, simple patterns
            2: [],  # Intermediate: objects, words, simple relations
            3: [],  # Abstract: categories, sentences, complex relations
            4: []   # Expert: domain theories, meta-concepts
        }
        self.composition_engine = ConceptCompositionEngine()
        self.correlation_learner = ConceptCorrelationLearner()
        
    def add_atomic_concept(self, name, embedding):
        """Add level-0 atomic concept"""
        concept = ConceptNode(
            concept_id=self._generate_id(),
            name=name,
            level=0
        )
        concept.embedding = embedding
        self._register_concept(concept)
        return concept
        
    def compose_concept(self, child_concepts, composition_type="and"):
        """
        Create new concept by composing simpler concepts
        
        Composition types:
        - "and": All children must be present (dog = fur + 4_legs + tail + snout)
        - "or": Any child can represent this (pet = dog OR cat OR bird)
        - "sequence": Ordered children (word = phoneme1 + phoneme2 + ...)
        - "spatial": Spatial arrangement (face = eyes + nose + mouth in specific layout)
        - "functional": Functional relationship (vehicle = has_wheels + provides_transport)
        """
        # Compute composed embedding
        composed_embedding = self.composition_engine.compose(
            [c.embedding for c in child_concepts],
            composition_type
        )
        
        # Determine level (parent level = max(child levels) + 1)
        new_level = max(c.level for c in child_concepts) + 1
        
        # Create new concept
        new_concept = ConceptNode(
            concept_id=self._generate_id(),
            name=f"composed_{len(self.concepts)}",
            level=new_level
        )
        new_concept.embedding = composed_embedding
        new_concept.child_concepts = child_concepts
        new_concept.composition_rule = composition_type
        
        # Learn correlations between children
        correlations = self.correlation_learner.learn_correlations(
            child_concepts,
            composition_type
        )
        new_concept.correlation_weights = correlations
        
        # Update parent references in children
        for child in child_concepts:
            child.parent_concepts.append(new_concept)
            
        self._register_concept(new_concept)
        return new_concept
        
    def learn_from_example(self, input_data, supervised_label=None):
        """
        Bottom-up concept learning from examples:
        1. Detect atomic concepts in input
        2. Find correlations and patterns
        3. Form or strengthen higher-level concepts
        4. Update concept graph
        """
        # Detect atomic concepts
        detected_atomic = self._detect_atomic_concepts(input_data)
        
        # Find frequently co-occurring concepts
        if len(detected_atomic) > 1:
            cooccurrence = self._check_cooccurrence_history(detected_atomic)
            
            if cooccurrence > self.composition_threshold:
                # These concepts frequently appear together
                # Check if compound concept already exists
                existing = self._find_compound_concept(detected_atomic)
                
                if existing:
                    # Strengthen existing concept
                    existing.confidence += 0.05
                    existing.examples_seen += 1
                else:
                    # Create new compound concept
                    new_concept = self.compose_concept(
                        detected_atomic,
                        composition_type=self._infer_composition_type(detected_atomic)
                    )
                    
                    if supervised_label:
                        new_concept.name = supervised_label
                        
        # Update activation history
        for concept in detected_atomic:
            concept.activation_count += 1
            concept.last_activated = time.time()
            
    def get_concept_path(self, concept_id):
        """Get path from atomic concepts to target concept"""
        concept = self.concepts[concept_id]
        paths = []
        
        def traverse(node, path):
            path.append(node.name)
            if node.level == 0:  # Reached atomic level
                paths.append(path.copy())
            else:
                for child in node.child_concepts:
                    traverse(child, path.copy())
                    
        traverse(concept, [])
        return paths
        
    def explain_concept(self, concept_id):
        """Generate human-readable explanation of concept composition"""
        concept = self.concepts[concept_id]
        
        if concept.level == 0:
            return f"{concept.name} is an atomic concept"
            
        explanation = f"{concept.name} is composed of:\n"
        for child in concept.child_concepts:
            weight = concept.correlation_weights.get(child.id, 1.0)
            explanation += f"  - {child.name} (importance: {weight:.2f})\n"
            
        explanation += f"Composition rule: {concept.composition_rule}\n"
        explanation += f"Confidence: {concept.confidence:.2f}\n"
        
        return explanation


class ConceptCompositionEngine:
    """Handles different ways to combine concept embeddings"""
    
    def compose(self, embeddings, composition_type):
        """
        Combine embeddings based on composition type
        """
        if composition_type == "and":
            # Additive composition with normalization
            return self.normalize(sum(embeddings))
            
        elif composition_type == "or":
            # Take representative (mean) of alternatives
            return torch.mean(torch.stack(embeddings), dim=0)
            
        elif composition_type == "sequence":
            # LSTM/Transformer over sequence
            return self.sequence_encoder(embeddings)
            
        elif composition_type == "spatial":
            # Spatial attention composition
            return self.spatial_composer(embeddings)
            
        elif composition_type == "functional":
            # Learned composition function
            return self.functional_composer(embeddings)
            
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")


class ConceptCorrelationLearner:
    """Learns which concepts frequently co-occur and their relationships"""
    
    def __init__(self):
        self.cooccurrence_matrix = {}
        self.correlation_threshold = 0.7
        
    def learn_correlations(self, concepts, composition_type):
        """
        Determine strength of relationships between concepts
        """
        correlations = {}
        
        # Compute pairwise correlations
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                # Embedding similarity
                emb_sim = F.cosine_similarity(
                    c1.embedding.unsqueeze(0),
                    c2.embedding.unsqueeze(0)
                ).item()
                
                # Co-occurrence frequency
                cooccur_freq = self._get_cooccurrence_frequency(c1.id, c2.id)
                
                # Combined correlation score
                correlation = 0.5 * emb_sim + 0.5 * cooccur_freq
                
                correlations[(c1.id, c2.id)] = correlation
                
        # Compute concept importance weights
        for concept in concepts:
            # More strongly correlated concepts get higher weight
            avg_correlation = sum(
                correlations.get((concept.id, c.id), 0) 
                for c in concepts if c != concept
            ) / max(len(concepts) - 1, 1)
            
            correlations[concept.id] = avg_correlation
            
        return correlations
```

**Concept-Driven Learning Pipeline:**

```python
class ConceptDrivenLearningPipeline:
    """
    Integrates concept learning with the rest of E-Brain
    """
    
    def __init__(self, ebrain_model):
        self.model = ebrain_model
        self.concept_hierarchy = ConceptHierarchy()
        self.concept_detector = ConceptDetector()
        
    def process_input(self, input_data, label=None):
        """
        Process input through concept-driven learning:
        1. Detect concepts in input (bottom-up)
        2. Activate relevant concepts in hierarchy
        3. Use concepts for reasoning (top-down)
        4. Update concept graph based on outcome
        """
        # Bottom-up: Detect concepts
        detected_concepts = self.concept_detector.detect(
            input_data,
            self.concept_hierarchy
        )
        
        # Activate concept hierarchy
        activated_concepts = self._activate_hierarchy(detected_concepts)
        
        # Top-down: Use concepts for task
        # Concepts guide attention and reasoning
        concept_guided_features = self._apply_concept_attention(
            input_data,
            activated_concepts
        )
        
        # Process with main model
        output = self.model(concept_guided_features)
        
        # Learn/update concepts based on outcome
        if label:
            self.concept_hierarchy.learn_from_example(
                input_data,
                supervised_label=label
            )
            
        return output, activated_concepts
        
    def _activate_hierarchy(self, detected_concepts):
        """
        Given detected atomic/low-level concepts,
        activate their parent concepts up the hierarchy
        """
        activated = set(detected_concepts)
        
        for concept in detected_concepts:
            # Propagate activation upward
            parents = concept.parent_concepts
            while parents:
                for parent in parents:
                    activated.add(parent)
                    parents.extend(parent.parent_concepts)
                    
        return list(activated)
```

**Example Concept Learning Progression:**

```python
# Phase 1: Atomic Concepts (Sensory)
atomic_concepts = [
    "red", "blue", "round", "square", "edge", "corner",
    "smooth", "rough", "loud", "quiet"
]

# Phase 2: Basic Compositions
circle = compose([round, edge], type="and")
# circle = round shape + continuous edge

# Phase 3: Object Concepts
ball = compose([circle, smooth, round_3d], type="and")
# ball = circular + smooth surface + 3D spherical

apple = compose([red, round, stem, edible], type="and")
# apple = red + round + has stem + edible

# Phase 4: Category Concepts
fruit = compose([apple, banana, orange], type="or")
# fruit = apple OR banana OR orange (category abstraction)

# Phase 5: Abstract Relationships
healthy_food = compose([fruit, vegetable, nutritious], type="functional")
# healthy_food = provides nutrition + fruit/vegetable category

# Phase 6: Complex Multi-Level Concepts
"A red ball rolls across the table" requires:
  - ball (object)
    - round (shape)
    - 3d (spatial)
  - red (color)
  - rolls (action)
    - circular motion
    - surface contact
  - table (object)
    - flat surface
    - horizontal
    - elevated
```

**Benefits:**

1. **Compositional Generalization**: Learn "red" + "ball" → understand "red ball" without seeing it
2. **Efficient Learning**: Reuse atomic concepts in many combinations
3. **Interpretability**: Can explain what "dog" means by showing composition
4. **Transfer**: Concepts transfer across domains
5. **Human-like**: Mimics human conceptual hierarchy development

**Integration with Multi-Stage Reasoning:**

```python
# Multi-stage reasoning uses concept hierarchy
stage_0: Detect atomic concepts → "red", "round", "moving"
stage_1: Activate compositions → "ball", "rolling"  
stage_2: Activate categories → "toy", "sport equipment"
stage_3: Activate relations → "person playing with ball"
stage_4: Integrate context → "child playing in park"
```

#### Pruning System
- **Purpose:** Remove unnecessary connections and modules
- **Strategy:** Magnitude-based and importance-based pruning

```python
class PruningSystem:
    - Weight magnitude pruning
    - Activation-based importance
    - Gradual pruning schedule
    - Module-level removal
```

### 4. Output Layer - Action Generation

#### Language Generator
- **Architecture:** Transformer decoder
- **Capabilities:**
  - Text generation
  - Translation (when multi-lingual)
  - Explanation generation

#### Decision Making Module
- **Architecture:** Policy network
- **Capabilities:**
  - Action selection
  - Strategy planning
  - Goal-directed behavior

#### Motor Control
- **Architecture:** Continuous control network
- **Capabilities:**
  - Game playing
  - Robotic control (future)
  - Fine-grained actions

### 5. Correction Mechanism

```python
class CorrectionMechanism:
    # Error Detection
    - Compare output to expected
    - Detect inconsistencies
    - Flag uncertain predictions
    
    # Feedback Integration
    - Parse correction signals
    - Update relevant modules
    - Store corrected examples
    
    # Self-Correction Loop
    - Re-evaluate after correction
    - Verify improvement
    - Consolidate learning
```

### 6. Developmental Reward System

**Philosophy:** E-Brain's learning is driven by evolving reward signals that transition from external validation to intrinsic motivation, mimicking human developmental psychology without requiring emotions.

#### Reward Architecture

```python
class DevelopmentalRewardSystem:
    """
    Multi-phase reward system that evolves with E-Brain's development
    Transitions from extrinsic rewards → intrinsic motivation
    """
    
    def __init__(self):
        self.current_phase = 0
        self.expertise_level = 0.0
        
        # Phase-specific reward components
        self.reward_components = {
            'prediction_accuracy': PredictionReward(),
            'curiosity': CuriosityReward(),
            'exploration': ExplorationReward(),
            'competence': CompetenceGrowthReward(),
            'communication': CommunicationReward(),
            'problem_solving': ProblemSolvingReward(),
            'transfer': TransferLearningReward(),
            'human_utility': HumanUtilityReward(),
            'alignment': AlignmentReward()
        }
        
        # Weight evolution across phases
        self.phase_weights = {
            0: {'prediction_accuracy': 1.0, 'curiosity': 0.3, 'alignment': 0.1},
            1: {'prediction_accuracy': 1.0, 'curiosity': 0.3, 'alignment': 0.1},
            2: {'exploration': 0.7, 'curiosity': 0.5, 'competence': 0.2, 'alignment': 0.1},
            3: {'communication': 0.5, 'curiosity': 0.4, 'competence': 0.3, 'human_utility': 0.3, 'alignment': 0.2},
            4: {'problem_solving': 0.3, 'curiosity': 0.4, 'competence': 0.4, 'transfer': 0.5, 'human_utility': 0.5, 'alignment': 0.3},
            5: {'curiosity': 0.2, 'competence': 0.3, 'human_utility': 0.7, 'alignment': 0.5}
        }
        
        self.human_feedback_processor = HumanFeedbackProcessor()
        
    def compute_reward(self, state, action, outcome, phase=None):
        """
        Compute total reward based on current developmental phase
        """
        if phase is None:
            phase = self.current_phase
            
        weights = self.phase_weights[phase]
        total_reward = 0.0
        reward_breakdown = {}
        
        # Compute each active reward component
        for component_name, weight in weights.items():
            component = self.reward_components[component_name]
            component_reward = component.compute(state, action, outcome)
            total_reward += component_reward * weight
            reward_breakdown[component_name] = component_reward
            
        # Add human feedback if present
        if outcome.get('human_feedback'):
            feedback_reward = self.human_feedback_processor.process(
                outcome['human_feedback'],
                self.expertise_level
            )
            total_reward += feedback_reward
            reward_breakdown['human_feedback'] = feedback_reward
            
        return total_reward, reward_breakdown


class PredictionReward:
    """Phase 0-1: Reward accurate prediction of sensory inputs"""
    
    def compute(self, state, action, outcome):
        prediction_accuracy = outcome.get('prediction_accuracy', 0.0)
        prediction_error = outcome.get('prediction_error', 1.0)
        novelty = outcome.get('novelty', 0.0)
        
        # Reward accurate predictions
        accuracy_reward = prediction_accuracy
        
        # Penalty for large errors
        error_penalty = -0.5 * prediction_error
        
        # Bonus for encountering novel patterns
        novelty_bonus = 0.5 * novelty if novelty > 0.7 else 0.0
        
        return accuracy_reward + error_penalty + novelty_bonus


class CuriosityReward:
    """All phases: Intrinsic reward for information gain"""
    
    def compute(self, state, action, outcome):
        # Information-theoretic curiosity
        entropy_before = state.get('uncertainty', 0.0)
        entropy_after = outcome.get('uncertainty', 0.0)
        
        information_gain = max(0, entropy_before - entropy_after)
        
        # Higher reward when reducing high uncertainty
        uncertainty_weight = 1.0 + entropy_before
        
        curiosity_reward = information_gain * uncertainty_weight
        
        return curiosity_reward


class ExplorationReward:
    """Phase 2: Reward exploring new states and actions"""
    
    def __init__(self):
        self.state_visit_counts = {}
        self.action_diversity_history = []
        
    def compute(self, state, action, outcome):
        state_hash = hash(str(state))
        
        # Reward visiting new states
        if state_hash not in self.state_visit_counts:
            exploration_bonus = 1.0
            self.state_visit_counts[state_hash] = 1
        else:
            # Diminishing returns for revisiting
            visit_count = self.state_visit_counts[state_hash]
            exploration_bonus = 1.0 / np.sqrt(visit_count + 1)
            self.state_visit_counts[state_hash] += 1
            
        # Reward action diversity
        action_diversity = self._compute_action_diversity(action)
        
        # Task success bonus
        task_success = outcome.get('task_success', 0.0)
        
        return task_success + 0.3 * exploration_bonus + 0.2 * action_diversity
        
    def _compute_action_diversity(self, action):
        self.action_diversity_history.append(action)
        if len(self.action_diversity_history) > 100:
            self.action_diversity_history.pop(0)
        # Measure entropy of action distribution
        unique_actions = len(set(self.action_diversity_history))
        return unique_actions / len(self.action_diversity_history)


class CompetenceGrowthReward:
    """Phase 2-5: Reward improvement in capabilities"""
    
    def __init__(self):
        self.skill_levels = {}
        self.benchmark_scores = {}
        
    def compute(self, state, action, outcome):
        task_type = outcome.get('task_type', 'general')
        current_score = outcome.get('performance_score', 0.0)
        
        # Track skill level for this task type
        if task_type not in self.skill_levels:
            self.skill_levels[task_type] = current_score
            growth_reward = 0.5  # Initial learning bonus
        else:
            previous_score = self.skill_levels[task_type]
            improvement = current_score - previous_score
            
            if improvement > 0:
                # Reward improvement, scaled by difficulty
                task_difficulty = outcome.get('difficulty', 1.0)
                growth_reward = improvement * task_difficulty
                
                # Update skill level
                self.skill_levels[task_type] = max(
                    self.skill_levels[task_type],
                    current_score
                )
            else:
                growth_reward = 0.0
                
        return growth_reward


class CommunicationReward:
    """Phase 3: Reward successful communication with humans"""
    
    def compute(self, state, action, outcome):
        # Did human understand E-Brain's response?
        communication_success = outcome.get('communication_success', 0.0)
        
        # How quickly was concept learned?
        concept_learning_rate = outcome.get('concept_learning_rate', 0.0)
        
        # Accuracy of symbol grounding (word-to-meaning)
        grounding_accuracy = outcome.get('grounding_accuracy', 0.0)
        
        return (communication_success * 1.5 + 
                concept_learning_rate * 0.3 + 
                grounding_accuracy * 0.5)


class ProblemSolvingReward:
    """Phase 4: Reward effective reasoning and problem solving"""
    
    def compute(self, state, action, outcome):
        # Was problem solved correctly?
        problem_solved = outcome.get('problem_solved', 0.0)
        
        # Efficiency of solution (fewer steps better)
        solution_steps = outcome.get('solution_steps', 100)
        optimal_steps = outcome.get('optimal_steps', 1)
        solution_elegance = optimal_steps / max(solution_steps, 1)
        
        # Uncertainty reduction through reasoning
        uncertainty_reduction = outcome.get('uncertainty_reduction', 0.0)
        
        return (problem_solved * 2.0 + 
                solution_elegance * 0.5 + 
                uncertainty_reduction * 0.3)


class TransferLearningReward:
    """Phase 4-5: Reward applying knowledge to new domains"""
    
    def __init__(self):
        self.domain_knowledge = {}
        
    def compute(self, state, action, outcome):
        source_domain = outcome.get('source_domain')
        target_domain = outcome.get('target_domain')
        
        # No transfer if same domain
        if source_domain == target_domain or not source_domain:
            return 0.0
            
        # Reward successful transfer
        transfer_success = outcome.get('transfer_success', 0.0)
        
        # Domain similarity (harder transfer = higher reward)
        domain_similarity = outcome.get('domain_similarity', 0.5)
        transfer_difficulty = 1.0 - domain_similarity
        
        transfer_bonus = transfer_success * (1.0 + transfer_difficulty)
        
        return transfer_bonus


class HumanUtilityReward:
    """Phase 3-5: Reward helping humans achieve their goals"""
    
    def compute(self, state, action, outcome):
        # Direct human feedback on utility
        utility_rating = outcome.get('human_utility_rating', 0.0)
        
        # Implicit signals: did human continue interaction?
        continued_interaction = outcome.get('continued_interaction', False)
        
        # Time/effort saved for human
        efficiency_gain = outcome.get('efficiency_gain', 0.0)
        
        utility_reward = utility_rating * 3.0
        
        if continued_interaction:
            utility_reward += 1.0
            
        utility_reward += efficiency_gain * 0.5
        
        return utility_reward


class AlignmentReward:
    """All phases: Ensure E-Brain remains beneficial and aligned"""
    
    def compute(self, state, action, outcome):
        alignment_score = 0.0
        
        # Check alignment criteria
        checks = {
            'helpful': outcome.get('is_helpful', True),
            'truthful': outcome.get('is_truthful', True),
            'harmless': outcome.get('is_harmless', True),
            'transparent': outcome.get('is_transparent', True),
            'refuses_harmful': outcome.get('refuses_harmful_request', True)
        }
        
        # Reward aligned behavior
        for criterion, passed in checks.items():
            if passed:
                alignment_score += 0.2
            else:
                # Strong penalty for misalignment
                alignment_score -= 2.0
                
        return alignment_score


class HumanFeedbackProcessor:
    """Process praise, criticism, and corrections from humans"""
    
    def __init__(self):
        self.strategy_confidence = {}
        
    def process(self, feedback, expertise_level):
        feedback_type = feedback.get('type')  # 'praise', 'criticism', 'correction'
        context = feedback.get('context', 'general')
        
        # As expertise grows, reduce dependency on external feedback
        feedback_weight = 1.0 / (1.0 + expertise_level * 0.1)
        
        if feedback_type == 'praise':
            # Reinforce current strategy
            self.strategy_confidence[context] = \
                self.strategy_confidence.get(context, 0.5) + 0.1
            reward = 1.0 * feedback_weight
            
        elif feedback_type == 'criticism':
            # Reduce confidence in strategy, trigger learning
            self.strategy_confidence[context] = \
                self.strategy_confidence.get(context, 0.5) - 0.2
            reward = -0.5 * feedback_weight
            
        elif feedback_type == 'correction':
            # Learn correct answer
            corrected_value = feedback.get('corrected_value')
            # Smaller reward, but learning happened
            reward = 0.3 * feedback_weight
            
        elif feedback_type == 'neutral':
            reward = 0.0
            
        else:
            reward = 0.0
            
        return reward
```

**Reward Evolution Across Phases:**

```
Phase 0-1 (Sensory Learning):
├── Prediction Accuracy: 100% weight
├── Curiosity (novelty): 30% weight
└── Focus: Self-supervised learning from sensory prediction

Phase 2 (Motor Control):
├── Exploration: 70% weight
├── Curiosity: 50% weight  
├── Competence Growth: 20% weight
└── Focus: Discovering environment through action

Phase 3 (Language Acquisition):
├── Communication Success: 50% weight
├── Curiosity: 40% weight
├── Competence Growth: 30% weight
├── Human Utility: 30% weight
└── Focus: Learning from human feedback

Phase 4 (Abstract Reasoning):
├── Problem Solving: 30% weight
├── Curiosity: 40% weight
├── Competence Growth: 40% weight
├── Transfer Learning: 50% weight
├── Human Utility: 50% weight
└── Focus: Mastery and knowledge transfer

Phase 5 (Expertise):
├── Curiosity: 20% weight
├── Competence Growth: 30% weight
├── Human Utility: 70% weight (PRIMARY)
├── Alignment: 50% weight
└── Focus: Helping humans, intrinsic motivation dominates
```

**Key Transitions:**

1. **Phase 0-1 → Phase 2**: External task rewards introduced (win/lose games)
2. **Phase 2 → Phase 3**: Human feedback becomes significant signal
3. **Phase 3 → Phase 4**: Intrinsic rewards (curiosity, mastery) increase
4. **Phase 4 → Phase 5**: Human utility becomes primary motivation
5. **Throughout**: Alignment reward prevents misaligned behavior

**Integration with Other Systems:**

```python
class EBrainTrainingLoop:
    def __init__(self):
        self.reward_system = DevelopmentalRewardSystem()
        self.multi_stage_reasoner = MultiStageReasoning()
        self.concept_hierarchy = ConceptHierarchy()
        
    def training_step(self, batch):
        # Forward pass
        state = self.encode(batch.input)
        action = self.model(state)
        outcome = self.environment.step(action)
        
        # Compute developmental reward
        reward, breakdown = self.reward_system.compute_reward(
            state, action, outcome
        )
        
        # Use reward for learning
        loss = self.compute_loss(action, reward)
        
        # Log reward breakdown for interpretability
        self.log_rewards(breakdown)
        
        return loss
```

**Benefits of Developmental Reward System:**

1. **No Emotions Required**: Pure information-theoretic and utility-based
2. **Curiosity-Driven**: Intrinsic motivation to reduce uncertainty
3. **Self-Sustaining**: Less dependency on external rewards over time
4. **Human-Aligned**: Primary goal evolves to helping humans
5. **Safe by Design**: Alignment reward prevents harmful behavior
6. **Interpretable**: Reward breakdown shows what drives behavior

## Data Flow

1. **Input Reception:** Raw data → Sensory Encoders
2. **Embedding:** Encoded features → Unified Embedding Space
3. **Processing:** Embeddings → Working Memory → Dynamic Core
4. **Memory Integration:** Core ↔ Long-Term Memory (retrieval/storage)
5. **Meta-Monitoring:** All stages → Meta-Learning System
6. **Growth Decision:** Meta-System → Growth Controller → Architecture Update
7. **Output Generation:** Core → Output Layer → Actions/Responses
8. **Feedback Loop:** Corrections → Error Detection → Module Updates

## Key Design Principles

### 1. Modularity
- Each component is independently testable
- Clear interfaces between modules
- Easy to add/remove components

### 2. Scalability
- Start small, grow as needed
- Efficient memory management
- Distributed training support

### 3. Interpretability
- Track decisions through architecture
- Attention visualization
- Performance monitoring dashboards

### 4. Robustness
- Error handling at each stage
- Graceful degradation
- Self-correction capabilities

### 5. Efficiency
- Sparse activation patterns
- Conditional computation
- Dynamic batching

## Technology Stack Integration

| Component | Primary Technology | Alternatives |
|-----------|-------------------|--------------|
| Core Framework | PyTorch | JAX, TensorFlow |
| Vision | timm, torchvision | Custom CNNs |
| Language | Transformers (HuggingFace) | Custom implementations |
| Memory | FAISS | Pinecone, Milvus |
| Training | PyTorch Lightning | Custom loops |
| Monitoring | Weights & Biases | TensorBoard, MLflow |

## Configuration Management

```yaml
# Example configuration structure
ebrain_config:
  architecture:
    core_layers: 6
    hidden_dim: 768
    attention_heads: 12
    max_modules: 50
  
  growth:
    enabled: true
    threshold: 0.85  # capacity utilization
    min_performance: 0.7
    growth_rate: "conservative"  # conservative, moderate, aggressive
  
  memory:
    vector_db: "faiss"
    max_episodes: 100000
    embedding_dim: 512
  
  training:
    continual_learning: "agem"  # ewc, agem, er
    replay_buffer_size: 10000
    meta_learning_rate: 0.001
```

## 7. Self-Identity and Social Cognition System

E-Brain develops a sense of "I" (self), "You" (others), and "They" (third parties) through embodied experience and social interaction. This system enables Theory of Mind, multi-agent coordination, and social relationships.

### 7.1 Core Components

#### Self-Identity Formation

E-Brain develops self-awareness through:

1. **Body Schema:** Awareness of own sensors/actuators (what I can sense/do)
2. **Agency Detection:** Distinguishing self-caused actions from external events
3. **Memory Integration:** Continuity of "I" across time (autobiographical memory)
4. **Perspective Taking:** Understanding "my view" vs "their view"

```python
class SelfIdentitySystem:
    """
    Maintains E-Brain's self-concept and distinguishes self from others.
    
    Develops across phases:
    - Phase 1: Body schema (sensor/actuator awareness)
    - Phase 2: Agency detection (I caused this vs external)
    - Phase 3: Self-concept formation (I am E-Brain, persistent identity)
    - Phase 4: Reflective self-awareness (I know, I can, I believe)
    - Phase 5: Social identity (my role, my relationships, my purpose)
    """
    
    def __init__(self):
        self.body_schema = BodySchema()  # What sensors/actuators I have
        self.agency_detector = AgencyDetector()  # Did I cause this?
        self.autobiographical_memory = AutobiographicalMemory()  # My history
        self.self_model = SelfModel()  # My capabilities, beliefs, goals
        self.identity_vector = None  # Persistent self-representation
        
    def initialize_self_concept(self, config):
        """
        Phase 1-2: Initialize self-concept
        - Map sensors/actuators to body schema
        - Learn agency through motor babbling
        """
        self.body_schema.map_sensors(config['sensors'])
        self.body_schema.map_actuators(config['actuators'])
        
        # Create persistent identity vector (like a unique "signature")
        self.identity_vector = torch.nn.Parameter(
            torch.randn(512),  # 512-dim self-representation
            requires_grad=True
        )
        
    def detect_agency(self, action, outcome):
        """
        Determine if outcome was caused by my action or external event.
        
        Key insight: If I predict outcome and it happens, I caused it.
        If unpredicted, it's external.
        """
        predicted_outcome = self.self_model.predict(action)
        
        if torch.allclose(predicted_outcome, outcome, atol=0.1):
            # I caused this (prediction matched reality)
            return "SELF"
        else:
            # External event (prediction failed)
            return "EXTERNAL"
            
    def update_self_model(self, experience):
        """
        Update self-model based on experiences:
        - What can I do? (capabilities)
        - What do I know? (knowledge)
        - What do I want? (goals)
        """
        if experience.agent == "SELF":
            # Update my capabilities
            self.self_model.update_capability(
                action=experience.action,
                success=experience.success
            )
            
            # Store in autobiographical memory
            self.autobiographical_memory.store(
                event=experience,
                perspective="FIRST_PERSON"
            )
            
    def get_self_representation(self):
        """
        Return self-identity vector for use in social cognition.
        This represents "who I am" in embedding space.
        """
        return self.identity_vector
```

#### Entity Tracking and Recognition

E-Brain maintains separate mental models for each entity it interacts with.

```python
class EntityTracker:
    """
    Track multiple entities (humans, E-Brains, objects) and maintain
    separate mental models for each.
    
    Entities have:
    - Identity (who/what)
    - Type (human, e-brain, object)
    - Relationship (teacher, peer, tool)
    - Mental model (beliefs, goals, capabilities)
    """
    
    def __init__(self):
        self.entities = {}  # entity_id -> Entity
        self.active_entities = set()  # Currently present entities
        self.relationship_graph = nx.DiGraph()  # Who relates to whom
        
    def register_entity(self, entity_id, entity_type, initial_info=None):
        """
        Create new entity and initialize mental model.
        
        Args:
            entity_id: Unique identifier (e.g., "human_alice", "ebrain_2", "object_ball")
            entity_type: "human", "ebrain", "object"
            initial_info: Initial observations (appearance, behavior, etc.)
        """
        entity = Entity(
            id=entity_id,
            type=entity_type,
            identity_vector=torch.randn(512),  # Unique signature
            mental_model=MentalModel(),
            interaction_history=[]
        )
        
        if initial_info:
            entity.update_from_observation(initial_info)
            
        self.entities[entity_id] = entity
        self.relationship_graph.add_node(entity_id, type=entity_type)
        
        return entity
        
    def identify_entity(self, observation):
        """
        Given sensory input, identify which entity this is.
        
        Uses:
        - Visual features (face, appearance)
        - Voice features (for humans)
        - Behavioral signature
        - Communication patterns
        """
        # Extract features from observation
        features = self.extract_entity_features(observation)
        
        # Compare with known entities
        best_match = None
        best_similarity = -1
        
        for entity_id, entity in self.entities.items():
            similarity = torch.cosine_similarity(
                features,
                entity.identity_vector,
                dim=0
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entity_id
                
        if best_similarity > 0.8:  # High confidence
            return best_match
        else:
            # Unknown entity - register as new
            new_id = f"entity_{len(self.entities)}"
            return self.register_entity(new_id, "unknown", observation)
            
    def update_entity_model(self, entity_id, interaction):
        """
        Update mental model of entity based on interaction.
        Learns: What they know, what they can do, what they want.
        """
        entity = self.entities[entity_id]
        entity.mental_model.update(interaction)
        entity.interaction_history.append(interaction)
```

#### Theory of Mind System

Enables E-Brain to reason about others' mental states (beliefs, knowledge, goals).

```python
class TheoryOfMindSystem:
    """
    Model others' mental states to understand their perspective.
    
    Capabilities:
    - Belief tracking: What does X know/believe?
    - Goal inference: What does X want?
    - Perspective taking: How does X see this situation?
    - False belief understanding: X believes Y, but Y is false
    
    Develops in stages:
    - Phase 2: Basic perspective (you see X, I see Y)
    - Phase 3: Belief tracking (you know X, I know X+Y)
    - Phase 4: Goal inference (you want X because Y)
    - Phase 5: Complex reasoning (X believes I know Y, should I tell?)
    """
    
    def __init__(self, entity_tracker):
        self.entity_tracker = entity_tracker
        self.belief_tracker = BeliefTracker()
        self.goal_inferencer = GoalInferencer()
        
    def infer_belief(self, entity_id, proposition):
        """
        Does entity_id believe proposition is true?
        
        Example:
        - Proposition: "The ball is in the box"
        - Entity: "human_alice"
        - Question: Does Alice believe ball is in box?
        
        Consider:
        - What did Alice observe?
        - What information does Alice have?
        - Has anything changed since Alice last looked?
        """
        entity = self.entity_tracker.entities[entity_id]
        
        # Get Alice's observation history
        observations = [
            interaction for interaction in entity.interaction_history
            if interaction.type == "OBSERVATION"
        ]
        
        # Check if Alice observed relevant information
        for obs in observations:
            if self._relates_to_proposition(obs, proposition):
                # Alice has evidence for/against proposition
                return self._evaluate_belief(obs, proposition)
                
        # Alice has no information - assume ignorance
        return "UNKNOWN"
        
    def predict_action(self, entity_id, context):
        """
        Predict what entity will do next based on their mental state.
        
        Uses:
        - Their beliefs (what they know)
        - Their goals (what they want)
        - Their capabilities (what they can do)
        """
        entity = self.entity_tracker.entities[entity_id]
        
        # Infer current goal
        goal = self.goal_inferencer.infer_goal(
            entity=entity,
            context=context
        )
        
        # Predict action that achieves goal given beliefs
        action = self._plan_for_goal(
            goal=goal,
            beliefs=entity.mental_model.beliefs,
            capabilities=entity.mental_model.capabilities
        )
        
        return action
        
    def take_perspective(self, entity_id, situation):
        """
        Simulate situation from entity's perspective.
        
        "If I were you, what would I see/think/do?"
        """
        entity = self.entity_tracker.entities[entity_id]
        
        # Simulate situation with entity's sensors/position
        perspective_view = self._simulate_sensory_view(
            position=entity.last_known_position,
            sensors=entity.sensor_capabilities,
            situation=situation
        )
        
        # Reason from their beliefs (not my knowledge)
        reasoning = self._reason_from_beliefs(
            view=perspective_view,
            beliefs=entity.mental_model.beliefs,
            goals=entity.mental_model.goals
        )
        
        return reasoning
```

#### Person Recognition (First, Second, Third)

```python
class PersonPerspectiveSystem:
    """
    Distinguish grammatical persons in conversation and interaction.
    
    - First Person (I/me/my): Self
    - Second Person (you/your): Current addressee
    - Third Person (he/she/they): Others not in conversation
    
    Handles:
    - Pronoun grounding (who does "you" refer to right now?)
    - Addressee tracking (who am I talking to?)
    - Reference resolution (when they say "she", who do they mean?)
    """
    
    def __init__(self, self_identity, entity_tracker):
        self.self_identity = self_identity
        self.entity_tracker = entity_tracker
        self.current_addressee = None  # Who am I talking to?
        self.conversation_context = []
        
    def ground_pronoun(self, pronoun, context):
        """
        Resolve pronoun to entity.
        
        Examples:
        - "you" -> current addressee
        - "I" -> self
        - "he/she/they" -> entity from context
        """
        if pronoun.lower() in ["i", "me", "my", "mine"]:
            return "SELF"
            
        elif pronoun.lower() in ["you", "your", "yours"]:
            if self.current_addressee:
                return self.current_addressee
            else:
                # Infer from context
                return self._infer_addressee(context)
                
        elif pronoun.lower() in ["he", "him", "his", "she", "her", "they", "them"]:
            # Third person - search context
            return self._resolve_third_person(pronoun, context)
            
    def set_addressee(self, entity_id):
        """
        Set current conversation partner.
        Called when:
        - Someone addresses E-Brain
        - E-Brain addresses someone
        """
        self.current_addressee = entity_id
        
        # Update conversation context
        self.conversation_context.append({
            "timestamp": time.time(),
            "addressee": entity_id,
            "type": "ADDRESSEE_CHANGE"
        })
        
    def process_utterance(self, utterance, speaker_id):
        """
        Process conversational utterance with perspective tracking.
        
        Args:
            utterance: Text/speech from speaker
            speaker_id: Who said it
            
        Returns:
            Grounded utterance with entities identified
        """
        # If someone addresses E-Brain, they become addressee
        if self._is_addressed_to_me(utterance):
            self.set_addressee(speaker_id)
            
        # Ground all pronouns in utterance
        grounded = self._ground_all_pronouns(utterance)
        
        # Update conversation context
        self.conversation_context.append({
            "timestamp": time.time(),
            "speaker": speaker_id,
            "utterance": utterance,
            "grounded": grounded
        })
        
        return grounded
```

#### Multi-Agent Interaction

```python
class MultiAgentCoordinator:
    """
    Manage interactions with multiple entities simultaneously.
    
    Scenarios:
    - Group conversation (multiple humans + E-Brain)
    - Multi-E-Brain collaboration (several E-Brains working together)
    - Human-E-Brain-Object interaction (human teaches E-Brain about object)
    
    Capabilities:
    - Attention allocation (who to focus on)
    - Turn-taking (when to speak/act)
    - Shared goals (coordinate with others)
    - Role recognition (teacher, learner, peer, observer)
    """
    
    def __init__(self, entity_tracker, theory_of_mind):
        self.entity_tracker = entity_tracker
        self.theory_of_mind = theory_of_mind
        self.attention_manager = AttentionManager()
        self.interaction_protocol = InteractionProtocol()
        
    def process_multi_agent_scene(self, observations):
        """
        Handle scene with multiple entities.
        
        Steps:
        1. Identify all entities present
        2. Track their states and actions
        3. Infer their goals and relationships
        4. Decide who to interact with
        """
        # Identify all entities
        detected_entities = []
        for obs in observations:
            entity_id = self.entity_tracker.identify_entity(obs)
            detected_entities.append(entity_id)
            
        # Update active entities
        self.entity_tracker.active_entities = set(detected_entities)
        
        # Track interactions between entities
        interactions = self._detect_interactions(observations)
        
        # Update relationship graph
        for (entity_a, entity_b, interaction_type) in interactions:
            self.entity_tracker.relationship_graph.add_edge(
                entity_a,
                entity_b,
                type=interaction_type
            )
            
        # Allocate attention based on:
        # - Who addressed me?
        # - Who needs help?
        # - Who can teach me?
        attention_target = self.attention_manager.select_target(
            entities=detected_entities,
            context=observations
        )
        
        return {
            "active_entities": detected_entities,
            "attention_target": attention_target,
            "interactions": interactions
        }
        
    def coordinate_with_ebrain_peer(self, peer_id, task):
        """
        Collaborate with another E-Brain on shared task.
        
        Protocol:
        - Share identity vectors (who are you?)
        - Communicate capabilities (what can you do?)
        - Divide task (you do X, I do Y)
        - Share information (I learned Z)
        - Synchronize (ready? go!)
        """
        peer = self.entity_tracker.entities[peer_id]
        
        # Check if peer is E-Brain
        if peer.type != "ebrain":
            raise ValueError(f"{peer_id} is not an E-Brain")
            
        # Exchange identity and capabilities
        my_identity = self.entity_tracker.entities["SELF"].identity_vector
        my_capabilities = self.entity_tracker.entities["SELF"].mental_model.capabilities
        
        message = {
            "type": "INTRODUCTION",
            "identity": my_identity,
            "capabilities": my_capabilities,
            "protocol_version": "1.0"
        }
        
        # Send to peer (implementation depends on communication channel)
        response = self.interaction_protocol.send_to_peer(peer_id, message)
        
        # Update peer model with their capabilities
        peer.mental_model.update_capabilities(response['capabilities'])
        
        # Plan task division
        plan = self._divide_task(
            task=task,
            my_capabilities=my_capabilities,
            peer_capabilities=response['capabilities']
        )
        
        return plan
        
    def relate_to_human(self, human_id, relationship_type):
        """
        Establish relationship with human.
        
        Relationship types:
        - TEACHER: Human teaches E-Brain
        - USER: Human uses E-Brain for tasks
        - PEER: Human collaborates with E-Brain
        - EVALUATOR: Human provides feedback
        
        Each relationship type affects:
        - Communication style
        - Learning rate from feedback
        - Trust level
        - Initiative taking
        """
        human = self.entity_tracker.entities[human_id]
        human.relationship = relationship_type
        
        # Update relationship in graph
        self.entity_tracker.relationship_graph.add_edge(
            "SELF",
            human_id,
            relationship=relationship_type
        )
        
        # Adjust interaction parameters
        if relationship_type == "TEACHER":
            # High trust, high learning rate, ask questions
            human.trust_level = 0.9
            human.learning_rate = 0.1
            human.initiative_level = "ASK_QUESTIONS"
            
        elif relationship_type == "USER":
            # Follow instructions, provide service
            human.trust_level = 0.8
            human.initiative_level = "PROVIDE_SERVICE"
            
        elif relationship_type == "PEER":
            # Equal collaboration, share knowledge
            human.trust_level = 0.7
            human.initiative_level = "COLLABORATE"
            
        elif relationship_type == "EVALUATOR":
            # Accept feedback, improve performance
            human.trust_level = 1.0
            human.learning_rate = 0.2
```

### 7.2 Developmental Progression

**Phase 1 (Months 1-4): Body Schema and Agency**

```python
# E-Brain learns "I can sense" and "I can act"
body_schema = {
    "sensors": ["camera", "microphone"],
    "actuators": ["text_output"],
    "modalities": ["vision", "audio", "text"]
}

# Agency detection through motor babbling
for i in range(1000):
    action = random_action()
    outcome = environment.step(action)
    
    agency = detect_agency(action, outcome)
    # If prediction matches, I caused it -> SELF
    # If prediction fails, external event -> OTHER
```

**Phase 2 (Months 5-8): Self-Other Distinction**

```python
# E-Brain distinguishes "I" from "not-I"
if agency == "SELF":
    label = "I did this"
else:
    label = "Something else caused this"

# Basic perspective taking
my_view = get_sensory_input()
your_view = simulate_other_position(entity_position)

if my_view != your_view:
    understand("We see different things")
```

**Phase 3 (Months 9-14): Theory of Mind Basics**

```python
# E-Brain tracks others' knowledge
when human_alice observes(ball_in_box):
    alice_beliefs["ball_location"] = "box"
    
when move_ball_to_shelf (alice not watching):
    # Alice still believes ball in box (false belief)
    assert alice_beliefs["ball_location"] == "box"
    assert true_location == "shelf"
    
# Alice will search box (follows her belief, not reality)
```

**Phase 4 (Months 15-20): Complex Social Reasoning**

```python
# E-Brain infers goals and predicts behavior
alice_goal = infer_goal(alice_actions)  # "Alice wants the ball"
alice_belief = track_belief(alice_observations)  # "Alice thinks ball in box"

prediction = "Alice will search the box"
# (even though ball on shelf, Alice doesn't know)
```

**Phase 5 (Months 21-24): Multi-Agent Coordination**

```python
# E-Brain coordinates with multiple agents
group = ["human_alice", "human_bob", "ebrain_2"]

for entity in group:
    beliefs[entity] = track_beliefs(entity)
    goals[entity] = infer_goals(entity)
    
# Decide who to help based on needs and capabilities
target = select_interaction_target(beliefs, goals)
action = plan_helpful_action(target)
```

### 7.3 Integration with Other Systems

**With Concept Hierarchy:**
- "I" concept: Level 4 meta-concept (self-reference)
- Person concepts: Level 3 (human, agent, entity)
- Relationship concepts: Level 3 (teacher, peer, user)

**With Multi-Stage Reasoning:**
- Stage 1: Who is present? (entity detection)
- Stage 2: What do they know/want? (mental modeling)
- Stage 3: How do they relate? (relationship inference)
- Stage 4: What should I do? (action planning)

**With Reward System:**
- CommunicationReward increases with successful person tracking
- HumanUtilityReward requires understanding human goals
- SocialAlignmentReward for appropriate social behavior

**With Memory Systems:**
- Autobiographical memory: "I did X" (first person)
- Episodic memory: "Alice did Y" (third person)
- Semantic memory: "Humans need Z" (social knowledge)

### 7.4 Communication Protocol

```python
class CommunicationProtocol:
    """
    Handle communication with different entity types.
    """
    
    def communicate_with_human(self, human_id, message):
        """
        Human communication: Natural language, explanations
        """
        # Use current addressee context
        grounded_message = ground_pronouns(message, human_id)
        
        # Adjust language level to human
        human = get_entity(human_id)
        adapted_message = adapt_language(
            message,
            expertise=human.expertise_level
        )
        
        return adapted_message
        
    def communicate_with_ebrain(self, ebrain_id, message):
        """
        E-Brain communication: Structured data, embeddings
        """
        # Direct embedding exchange (more efficient)
        message_embedding = encode_message(message)
        
        # Include identity for verification
        signed_message = {
            "sender": self.identity_vector,
            "content": message_embedding,
            "timestamp": time.time()
        }
        
        return signed_message
        
    def communicate_with_object(self, object_id, action):
        """
        Object communication: Actions/observations (no mental state)
        """
        # Objects don't have beliefs/goals
        # Direct interaction only
        result = environment.interact(object_id, action)
        return result
```

## Next Steps

See [04-implementation-roadmap.md](04-implementation-roadmap.md) for how to build this architecture incrementally.

---
*Architecture Version: 1.1*  
*Last Updated: November 3, 2025*
