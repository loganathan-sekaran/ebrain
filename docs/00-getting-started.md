# E-Brain Development Checklist

## 🎯 Quick Start: Your First Week

### Day 1: Setup & Understanding
- [ ] Read [05-development-strategy.md](05-development-strategy.md) - Understand implementation vs training
- [ ] Read [02-architecture.md](02-architecture.md) sections 1-3 - Core neuron design
- [ ] Set up development environment:
  ```bash
  git clone https://github.com/loganathan-sekaran/ebrain.git
  cd ebrain
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Verify PyTorch installation:
  ```python
  import torch
  print(f"PyTorch version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")
  ```

### Day 2-3: First Component - BioInspiredNeuron
- [ ] Create `src/core/neuron.py`
- [ ] Implement `BioInspiredNeuron` class:
  ```python
  class BioInspiredNeuron:
      def __init__(self, neuron_id, threshold=1.0):
          self.id = neuron_id
          self.threshold = threshold
          self.membrane_potential = 0.0
          self.dendrites = []  # Input connections
          self.spike_times = []  # History of spikes
      
      def add_dendrite(self, source_neuron, initial_weight=0.5):
          """Add input connection from another neuron"""
          pass
      
      def integrate_inputs(self, inputs, dt=0.001):
          """Leaky integration of inputs"""
          pass
      
      def check_spike(self):
          """Returns True if membrane potential > threshold"""
          pass
      
      def apply_stdp(self, pre_spike_time, post_spike_time):
          """Spike-timing-dependent plasticity"""
          pass
  ```
- [ ] Write tests in `tests/test_neuron.py`:
  ```python
  def test_neuron_spikes_when_above_threshold():
      neuron = BioInspiredNeuron(neuron_id=1, threshold=1.0)
      neuron.membrane_potential = 1.5
      assert neuron.check_spike() == True
  
  def test_stdp_strengthens_causal_connections():
      # Pre-spike at t=100ms, post-spike at t=120ms
      # Weight should increase (pre causes post)
      pass
  ```
- [ ] Run tests: `pytest tests/test_neuron.py -v`

### Day 4-5: Vision Input System
- [ ] Create `src/sensory/vision.py`
- [ ] Implement basic vision encoder:
  ```python
  class VisionInputSystem:
      def __init__(self, resolution=(64, 64)):
          self.resolution = resolution
      
      def preprocess_image(self, image):
          """Resize, normalize image"""
          pass
      
      def encode_to_spikes(self, image):
          """Convert pixel intensities to spike rates"""
          # Bright pixels → high spike rate
          # Dark pixels → low spike rate
          pass
  ```
- [ ] Test with simple patterns
- [ ] Verify spike encoding works

### Day 6-7: First Training Data
- [ ] Create data generator in `scripts/generate_poc_data.py`:
  ```python
  def generate_moving_square_data(num_frames=10000):
      """Generate simple moving square for POC"""
      data = []
      for i in range(num_frames):
          frame = np.zeros((64, 64))
          x_pos = (i % 60)
          frame[30:34, x_pos:x_pos+4] = 1.0
          
          next_frame = np.zeros((64, 64))
          next_x = ((i + 1) % 60)
          next_frame[30:34, next_x:next_x+4] = 1.0
          
          data.append({
              'current': frame,
              'next': next_frame
          })
      return data
  ```
- [ ] Generate and save data: `python scripts/generate_poc_data.py`
- [ ] Verify data looks correct (visualize a few frames)

## 📋 Month 1-2: Core Infrastructure

### Week 1-2: Neuron Implementation
- [ ] Complete `BioInspiredNeuron` class
- [ ] Implement STDP learning rule
- [ ] Add spike timing tracking
- [ ] Write comprehensive unit tests
- [ ] Document neuron parameters

### Week 3-4: Neurogenesis System
- [ ] Create `src/core/neurogenesis.py`
- [ ] Implement `NeurogenesisSystem` class:
  - [ ] `add_neuron()` - Create new neurons
  - [ ] `prune_connection()` - Remove weak synapses
  - [ ] `hebbian_rewiring()` - Strengthen active pathways
- [ ] Test neuron growth mechanisms
- [ ] Test pruning mechanisms

### Week 5-6: Vision System
- [ ] Expand `VisionInputSystem`:
  - [ ] Edge detection (Gabor filters)
  - [ ] Corner detection
  - [ ] Basic feature extraction
- [ ] Test on MNIST subset
- [ ] Benchmark performance

### Week 7-8: Reward System
- [ ] Create `src/learning/reward_system.py`
- [ ] Implement `NoveltyDetector` class
- [ ] Implement `PredictionTracker` class
- [ ] Implement `RewardSystem` class
- [ ] Test reward computation

### Week 8: Integration Testing
- [ ] Connect all components together
- [ ] Test full forward pass (vision → neurons → output)
- [ ] Verify STDP updates weights
- [ ] Document integration points
- [ ] **DELIVERABLE**: Functional E-Brain codebase (untrained)

## 🎓 Month 3: Proof of Concept Training

### Week 1: Training Setup
- [ ] Create `training/train_poc.py`
- [ ] Implement training loop:
  ```python
  def train_poc():
      ebrain = EBrainSystem(initial_neurons=100)
      data = load_generated_data()
      
      for epoch in range(50):
          for sample in data:
              # Forward pass
              prediction = ebrain.forward(sample['current'])
              
              # Compute reward
              error = mse(prediction, sample['next'])
              reward = -error + novelty_bonus
              
              # Learn
              ebrain.learn(reward)
          
          # Evaluate
          accuracy = evaluate(ebrain, test_data)
          print(f"Epoch {epoch}: Accuracy {accuracy:.2%}")
      
      ebrain.save("checkpoints/poc.pt")
  ```
- [ ] Set up logging (TensorBoard or Weights & Biases)
- [ ] Define evaluation metrics

### Week 2-3: Run POC Training
- [ ] Launch training: `python training/train_poc.py`
- [ ] Monitor training progress
- [ ] Save checkpoints every 10 epochs
- [ ] Track metrics:
  - [ ] Prediction accuracy
  - [ ] Number of neurons (should grow)
  - [ ] Connection count
  - [ ] Reward trend

### Week 4: Evaluation & Documentation
- [ ] Run comprehensive evaluation suite
- [ ] Verify success criteria:
  - [ ] >70% prediction accuracy ✓
  - [ ] Neurons grew 100 → 500+ ✓
  - [ ] STDP learning occurred ✓
- [ ] Document findings
- [ ] Prepare Phase 1 implementation plan
- [ ] **DELIVERABLE**: `checkpoints/poc.pt` - First trained E-Brain!

## 📊 Progress Tracking

### Completed When:
```
✅ All unit tests pass
✅ Integration tests pass
✅ Code is documented
✅ Component works as expected
```

### Ready for Next Phase When:
```
✅ Current phase success criteria met
✅ Checkpoint saved and validated
✅ Documentation updated
✅ Team review completed
```

## 🚨 Common Pitfalls to Avoid

### ❌ Don't Do This:
- Start training before implementation is complete
- Skip unit tests ("I'll add them later")
- Train without defining success criteria
- Change architecture during training
- Debug data loaders during expensive GPU time
- Train without saving checkpoints

### ✅ Do This Instead:
- Implement → Test → Train (in that order)
- Write tests for each component
- Define metrics before training starts
- Freeze architecture during training phase
- Validate data pipeline before GPU time
- Save checkpoints every epoch

## 📈 Success Metrics by Phase

### POC (Month 3)
- [ ] Prediction accuracy >70%
- [ ] Neurons grow from 100 → 500+
- [ ] Training completes in <4 hours
- [ ] Model checkpoint saves successfully

### Phase 1 (Month 6)
- [ ] MNIST accuracy >85%
- [ ] Edge detection works
- [ ] Agency detection >80% accurate
- [ ] 50+ concepts grounded

### Phase 2-3 (Month 10)
- [ ] 1000+ word vocabulary
- [ ] COCO grounding >80% accurate
- [ ] Theory of Mind >80% accurate
- [ ] 4 concurrent thoughts working

### Phase 4-5 (Month 20)
- [ ] RAVEN accuracy >70%
- [ ] GSM8K accuracy >80%
- [ ] 7 concurrent thoughts working
- [ ] Expert-level domain performance

## 🛠️ Development Tools

### Required Tools
- [ ] Python 3.10+
- [ ] PyTorch 2.0+
- [ ] CUDA toolkit (for GPU)
- [ ] Git
- [ ] pytest (testing)
- [ ] TensorBoard or W&B (monitoring)

### Recommended Tools
- [ ] VS Code with Python extension
- [ ] Jupyter notebooks (for exploration)
- [ ] Docker (for reproducibility)
- [ ] tmux/screen (long-running jobs)

### Hardware Requirements
- **Minimum**: CPU-only, 16GB RAM (for development)
- **Recommended**: 1x GPU (RTX 3090 or better), 32GB RAM
- **Ideal**: 2x GPU, 64GB RAM (for Phase 4-5 training)

## 📚 Learning Resources

### Before You Start
1. Read: "Neuroscience for AI" - Understand bio-inspired computing
2. Watch: PyTorch tutorials - Master the framework
3. Review: STDP learning papers - Understand the learning rule
4. Study: Existing work (Nengo, Spaun) - Learn from others

### During Development
- Keep [02-architecture.md](02-architecture.md) open as reference
- Refer to [05-development-strategy.md](05-development-strategy.md) for workflow
- Check [04-implementation-roadmap.md](04-implementation-roadmap.md) for milestones
- Use [05-development-strategy-visual.md](05-development-strategy-visual.md) for timeline

## 🎯 Your First Month Goals

By end of Month 1, you should have:
1. ✅ Development environment set up
2. ✅ `BioInspiredNeuron` class implemented and tested
3. ✅ `NeurogenesisSystem` class implemented and tested
4. ✅ `VisionInputSystem` class implemented and tested
5. ✅ `RewardSystem` class implemented and tested
6. ✅ All unit tests passing
7. ✅ Integration tests passing
8. ✅ POC training data generated
9. ✅ Training script ready to run
10. ✅ Team ready for first training run!

## 🚀 Ready to Start?

```bash
# Create your feature branch
git checkout -b feature/bio-inspired-neuron

# Start coding!
mkdir -p src/core
touch src/core/neuron.py
touch tests/test_neuron.py

# Open your editor
code src/core/neuron.py
```

**Remember**: Build the brain first, then train it! 🧠

Good luck! 🎉
