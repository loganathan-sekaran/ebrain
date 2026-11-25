# Appendix A: Bio-Inspired Neuron (Research Branch)

> **Status: research branch. Not on the critical path.**
>
> This material was §0 of [02-architecture.md](02-architecture.md) until November 2026. It was moved here under TRK-14 of the [November 2026 design review](13-design-review-2026-11.md).
>
> The canonical plan ([PLAN.md §5](PLAN.md#5-decisions-on-record)) forbids per-neuron Python objects. All mechanisms on the critical path are implemented as vectorized PyTorch tensor operations; see [02 §0.1](02-architecture.md#01-implementation-strategy-biological-inspiration-vs-vectorized-deep-learning).
>
> Treat the code below as *conceptual pseudocode* for a possible spiking/STDP research track. **Do not implement it as written**:
> - `BioInspiredLayer` loops over one `nn.Module` per neuron, which is orders of magnitude slower than batched ops.
> - Discrete spikes need surrogate gradients (e.g. snnTorch, SpikingJelly).
> - The per-calendar-month growth timeline and the prune-threshold comments in §A.8 are not valid design rules.
>
> Horizon 1 growth is specified in [14-horizon-1-spec.md §4](14-horizon-1-spec.md#4-growth-mechanism).
>
> If this branch is pursued, it needs its own spec: vectorized LIF layers with surrogate gradients, and a benchmark where temporal dynamics matter. It also needs its own go/no-go gate, independent of Horizons 1–3.

## A.0 Bio-Inspired Neuron

Before diving into system architecture, we define the fundamental building block: an enhanced artificial neuron that mimics key biological properties.

### A.1 Biological Neurons vs Traditional Artificial Neurons

**Traditional Artificial Neuron (Standard Deep Learning):**
```python
# Simple weighted sum + activation
output = activation(sum(weights * inputs) + bias)

# Limitations:
# - No temporal dynamics (instantaneous computation)
# - No local learning (requires global backpropagation)
# - No dendritic computation (single weighted sum)
# - No spike timing (continuous values, no discrete spikes)
# - No plasticity during inference (weights frozen after training)
# - No neuromodulation (fixed learning rate)
```

**Biological Neuron Properties We Want to Mimic:**

1. **Temporal Dynamics**: Neurons integrate inputs over time, have refractory periods, generate discrete spikes
2. **Dendritic Computation**: Dendrites perform local nonlinear computations before reaching soma
3. **Synaptic Plasticity**: Connections strengthen/weaken based on correlated activity (Hebbian learning, STDP)
4. **Neuromodulation**: Global signals (dopamine, serotonin) modulate learning rates and connectivity
5. **Sparse Activation**: Most neurons inactive most of the time (energy efficient)
6. **Local Learning Rules**: Weight updates based on local pre/post-synaptic activity, not global error signals
7. **Multiple Timescales**: Fast spikes (milliseconds) to slow structural changes (hours/days)
8. **Homeostatic Plasticity**: Neurons maintain target firing rates, self-regulate excitability

### A.2 Enhanced Bio-Inspired Neuron Design

Our neuron design balances biological realism with computational practicality:

```python
import torch
import torch.nn as nn
import numpy as np

class BioInspiredNeuron(nn.Module):
    """
    Enhanced artificial neuron mimicking key biological properties.
    
    Components:
    1. Dendritic branches: Local nonlinear computation
    2. Soma: Integration and spike generation
    3. Axon: Output transmission with temporal dynamics
    4. Synapses: Plastic connections with local learning rules
    5. Neuromodulation: Global learning rate modulation
    
    Key Features:
    - Temporal dynamics (leaky integration)
    - Spike generation (optional, can be continuous or spiking)
    - Local plasticity (STDP, Hebbian learning)
    - Homeostatic regulation
    - Sparse activation
    """
    
    def __init__(
        self,
        input_dim: int,
        num_dendrites: int = 4,
        dendrite_dim: int = None,
        activation: str = 'relu',
        spiking: bool = False,
        spike_threshold: float = 1.0,
        leak_rate: float = 0.9,
        refractory_period: int = 2,
        enable_stdp: bool = True,
        enable_homeostasis: bool = True,
        target_firing_rate: float = 0.1
    ):
        super().__init__()
        
        # Dendritic branches
        self.num_dendrites = num_dendrites
        self.dendrite_dim = dendrite_dim or (input_dim // num_dendrites)
        
        # Each dendrite gets a subset of inputs and does local computation
        self.dendrites = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dendrite_dim, self.dendrite_dim),
                nn.ReLU(),  # Local nonlinearity
                nn.Linear(self.dendrite_dim, self.dendrite_dim)
            )
            for _ in range(num_dendrites)
        ])
        
        # Soma integration
        total_dendrite_output = num_dendrites * self.dendrite_dim
        self.soma_weights = nn.Linear(total_dendrite_output, 1, bias=True)
        
        # Temporal dynamics
        self.leak_rate = leak_rate  # How fast membrane potential decays
        self.membrane_potential = 0.0  # Current potential
        self.register_buffer('voltage_history', torch.zeros(100))  # Track history
        
        # Spiking mechanism
        self.spiking = spiking
        self.spike_threshold = spike_threshold
        self.refractory_period = refractory_period
        self.refractory_counter = 0
        self.last_spike_time = -1000  # Initialize far in past
        
        # Plasticity
        self.enable_stdp = enable_stdp
        self.stdp_lr = 0.001  # Local learning rate
        self.trace_decay = 0.95  # Eligibility trace decay
        
        # Store pre-synaptic traces for STDP
        self.register_buffer('pre_trace', torch.zeros(input_dim))
        self.register_buffer('post_trace', torch.tensor(0.0))
        
        # Homeostatic plasticity
        self.enable_homeostasis = enable_homeostasis
        self.target_firing_rate = target_firing_rate
        self.register_buffer('actual_firing_rate', torch.tensor(0.0))
        self.homeostatic_scale = 1.0
        
        # Neuromodulation (set externally)
        self.neuromodulation_signal = 1.0  # Default no modulation
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Sparsity
        self.activity_count = 0
        self.total_steps = 0
        
    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x, timestep=0):
        """
        Forward pass with temporal dynamics.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            timestep: Current time step (for temporal dynamics)
            
        Returns:
            output: Neuron output (spike or continuous value)
            auxiliary: Dict with additional info (potential, spike, etc.)
        """
        batch_size = x.shape[0]
        
        # 1. DENDRITIC COMPUTATION
        # Split input across dendritic branches
        branch_size = x.shape[1] // self.num_dendrites
        dendrite_outputs = []
        
        for i, dendrite in enumerate(self.dendrites):
            start_idx = i * branch_size
            end_idx = start_idx + branch_size if i < self.num_dendrites - 1 else x.shape[1]
            branch_input = x[:, start_idx:end_idx]
            
            # Pad if necessary
            if branch_input.shape[1] < self.dendrite_dim:
                padding = torch.zeros(
                    batch_size, 
                    self.dendrite_dim - branch_input.shape[1],
                    device=x.device
                )
                branch_input = torch.cat([branch_input, padding], dim=1)
            
            # Local dendritic computation
            dendrite_out = dendrite(branch_input)
            dendrite_outputs.append(dendrite_out)
        
        # Combine dendritic outputs
        dendrite_combined = torch.cat(dendrite_outputs, dim=1)
        
        # 2. SOMA INTEGRATION
        # Compute somatic current
        somatic_current = self.soma_weights(dendrite_combined).squeeze(-1)
        
        # Apply homeostatic scaling
        somatic_current = somatic_current * self.homeostatic_scale
        
        # 3. TEMPORAL DYNAMICS
        # Leaky integration (simplified leaky integrate-and-fire)
        if isinstance(self.membrane_potential, float):
            self.membrane_potential = torch.zeros(batch_size, device=x.device)
        
        # Decay existing potential
        self.membrane_potential = self.leak_rate * self.membrane_potential + somatic_current
        
        # 4. SPIKE GENERATION or CONTINUOUS OUTPUT
        if self.spiking:
            # Spiking neuron
            spike = torch.zeros_like(self.membrane_potential)
            
            # Check refractory period
            if self.refractory_counter <= 0:
                # Generate spike if above threshold
                spike_mask = self.membrane_potential > self.spike_threshold
                spike[spike_mask] = 1.0
                
                # Reset membrane potential where spike occurred
                self.membrane_potential[spike_mask] = 0.0
                
                # Set refractory period
                if spike.sum() > 0:
                    self.refractory_counter = self.refractory_period
                    self.last_spike_time = timestep
            else:
                self.refractory_counter -= 1
            
            output = spike
            
            # Update firing rate
            if self.enable_homeostasis:
                self.actual_firing_rate = 0.99 * self.actual_firing_rate + 0.01 * spike.mean()
        else:
            # Continuous neuron (standard)
            output = self.activation(self.membrane_potential)
        
        # 5. UPDATE PLASTICITY TRACES
        if self.training and self.enable_stdp:
            # Update pre-synaptic trace (input)
            self.pre_trace = self.trace_decay * self.pre_trace + x.mean(0)
            
            # Update post-synaptic trace (output)
            self.post_trace = self.trace_decay * self.post_trace + output.mean()
        
        # 6. TRACK SPARSITY
        self.total_steps += 1
        self.activity_count += (output > 0).float().mean().item()
        
        # Auxiliary information
        auxiliary = {
            'membrane_potential': self.membrane_potential.clone(),
            'spike': output if self.spiking else None,
            'dendrite_outputs': dendrite_outputs,
            'sparsity': self.activity_count / max(self.total_steps, 1),
            'firing_rate': self.actual_firing_rate.item() if self.spiking else None
        }
        
        return output, auxiliary
    
    def apply_stdp(self, reward_signal=1.0):
        """
        Apply Spike-Timing-Dependent Plasticity (STDP).
        
        Hebbian rule: Neurons that fire together, wire together.
        - If pre fires before post: strengthen (LTP - Long-Term Potentiation)
        - If post fires before pre: weaken (LTD - Long-Term Depression)
        
        Modulated by reward signal (reward-modulated STDP).
        """
        if not self.enable_stdp or not self.training:
            return
        
        # STDP update for soma weights
        # Δw = η * (post_trace * pre_trace) * reward * neuromodulation
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'soma_weights.weight' in name:
                    # Hebbian update: correlate pre and post activity
                    # Outer product would be ideal, but simplified here
                    stdp_update = (
                        self.stdp_lr * 
                        self.post_trace * 
                        reward_signal *
                        self.neuromodulation_signal
                    )
                    
                    # Apply update (simplified - would need proper pre-post correlation)
                    param.data += stdp_update * torch.randn_like(param) * 0.01
    
    def apply_homeostasis(self):
        """
        Homeostatic plasticity: Maintain target firing rate.
        
        If firing too much: decrease excitability (scale down weights)
        If firing too little: increase excitability (scale up weights)
        """
        if not self.enable_homeostasis or not self.spiking:
            return
        
        # Adjust homeostatic scale based on firing rate
        rate_error = self.target_firing_rate - self.actual_firing_rate.item()
        
        # Slowly adjust scale (slow homeostatic timescale)
        self.homeostatic_scale += 0.0001 * rate_error
        self.homeostatic_scale = np.clip(self.homeostatic_scale, 0.5, 2.0)
    
    def set_neuromodulation(self, signal: float):
        """
        Set neuromodulation signal (e.g., dopamine level).
        Affects learning rate and plasticity.
        
        signal > 1.0: Enhanced learning (reward)
        signal < 1.0: Reduced learning (punishment)
        signal = 1.0: Baseline
        """
        self.neuromodulation_signal = signal
    
    def reset_state(self):
        """Reset temporal state (membrane potential, traces, etc.)"""
        self.membrane_potential = 0.0
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.refractory_counter = 0
        self.last_spike_time = -1000
    
    def get_sparsity(self):
        """Get activation sparsity (fraction of time neuron is active)"""
        return self.activity_count / max(self.total_steps, 1)
```

### A.3 Key Features Explained

#### **1. Dendritic Computation**
```python
# Traditional: Single weighted sum
output = activation(W @ x + b)

# Bio-inspired: Multiple dendritic branches with local computation
dendrite_1 = nonlinear(W1 @ x[0:n])
dendrite_2 = nonlinear(W2 @ x[n:2n])
# ... combine at soma
```

**Why:** Dendrites in real neurons compute locally before reaching soma, enabling more complex input integration patterns.

#### **2. Temporal Dynamics (Leaky Integration)**
```python
# Membrane potential decays over time
V(t) = leak_rate * V(t-1) + input_current(t)

# Biological: τ dV/dt = -V + I
# Discrete: V[t] = α*V[t-1] + I[t]
```

**Why:** Real neurons integrate inputs over time, not instantaneously. Enables temporal coding and sequence learning.

#### **3. Spike-Timing-Dependent Plasticity (STDP)**
```python
# Hebbian learning with timing
if pre_spike_before_post:
    Δw = +A * exp(-(t_post - t_pre)/τ)  # LTP: strengthen
elif post_spike_before_pre:
    Δw = -A * exp(-(t_pre - t_post)/τ)  # LTD: weaken
```

**Why:** Causality matters - if presynaptic spike causes postsynaptic spike, strengthen connection. Core of biological learning.

#### **4. Homeostatic Plasticity**
```python
# Maintain target firing rate
if actual_rate > target_rate:
    scale_down_excitability()
elif actual_rate < target_rate:
    scale_up_excitability()
```

**Why:** Prevents runaway excitation or silence. Neurons self-regulate to maintain useful activity levels.

#### **5. Neuromodulation**
```python
# Global signal modulates learning
learning_rate_effective = base_lr * neuromodulation_signal

# Examples:
# dopamine_high (reward) → learn faster
# dopamine_low (punishment) → learn slower
```

**Why:** Enables reward-modulated learning, attention, motivation - critical for reinforcement learning.

### A.4 Usage in E-Brain Architecture

**Layer Construction:**
```python
class BioInspiredLayer(nn.Module):
    """Layer of bio-inspired neurons"""
    
    def __init__(self, input_dim, output_dim, neuron_config):
        super().__init__()
        self.neurons = nn.ModuleList([
            BioInspiredNeuron(input_dim, **neuron_config)
            for _ in range(output_dim)
        ])
    
    def forward(self, x, timestep=0):
        outputs = []
        auxiliaries = []
        
        for neuron in self.neurons:
            out, aux = neuron(x, timestep)
            outputs.append(out)
            auxiliaries.append(aux)
        
        return torch.stack(outputs, dim=1), auxiliaries
    
    def apply_plasticity(self, reward_signal):
        """Apply local learning rules across all neurons"""
        for neuron in self.neurons:
            neuron.apply_stdp(reward_signal)
            neuron.apply_homeostasis()
```

**Progressive Complexity:**
- **Phase 1**: Use continuous mode (spiking=False) - similar to standard neurons, easier training
- **Phase 2-3**: Enable temporal dynamics (leak_rate < 1.0) - learn sequences
- **Phase 4**: Enable STDP (enable_stdp=True) - local learning alongside backprop
- **Phase 5**: Full spiking mode (spiking=True) - maximum biological realism

### A.5 Comparison Table

| Feature | Traditional Neuron | Bio-Inspired Neuron | Biological Neuron |
|---------|-------------------|---------------------|-------------------|
| Computation | Weighted sum + activation | Dendritic branches + soma | Dendritic computation + soma |
| Temporal | Instantaneous | Leaky integration | Complex dynamics |
| Output | Continuous value | Continuous or spikes | Action potentials (spikes) |
| Learning | Backpropagation (global) | Backprop + STDP (local) | STDP, LTP, LTD (local) |
| Plasticity | Training phase only | Online (continuous) | Continuous lifelong |
| Homeostasis | None | Target firing rate | Multiple mechanisms |
| Neuromodulation | Learning rate (manual) | Signal-based (automatic) | Neurotransmitters (dopamine, etc.) |
| Sparsity | Dense activation | Tracked and optimized | Naturally sparse (~1%) |

### A.6 Advantages for E-Brain

1. **Temporal Learning**: Better at sequences, time series, temporal credit assignment
2. **Local Plasticity**: Can learn online during inference, not just training
3. **Energy Efficiency**: Sparse activation reduces computation (like real brain)
4. **Biological Plausibility**: More interpretable, closer to neuroscience
5. **Neuromodulation**: Natural integration with reward system (dopamine-like signals)
6. **Homeostasis**: Self-regulating, prevents overfitting or dead neurons
7. **Developmental Growth**: Easier to add/remove neurons dynamically

### A.7 Implementation Strategy

**Hybrid Approach** (Practical for E-Brain):

```python
# Start simple, add complexity over developmental phases
config_phase1 = {
    'num_dendrites': 1,  # Simple (essentially traditional neuron)
    'spiking': False,
    'enable_stdp': False,
    'leak_rate': 1.0  # No temporal dynamics yet
}

config_phase3 = {
    'num_dendrites': 4,  # Add dendritic computation
    'spiking': False,  # Still continuous for compatibility
    'enable_stdp': True,  # Enable local learning
    'leak_rate': 0.9  # Add temporal dynamics
}

config_phase5 = {
    'num_dendrites': 8,  # Full dendritic computation
    'spiking': True,  # Full spiking mode
    'enable_stdp': True,
    'enable_homeostasis': True,
    'leak_rate': 0.95
}
```

**Trade-offs:**
- **More biological = More computation**: Spiking neurons are slower than continuous
- **Local learning = Less efficient**: STDP trains slower than backprop
- **Solution**: Use hybrid (backprop for fast learning, STDP for online adaptation)

This bio-inspired neuron forms the foundation for E-Brain's neural architecture, enabling biological realism while maintaining computational practicality.

### A.8 Neuron Growth, Pruning, and Reorganization

E-Brain's neurons can be dynamically spawned, pruned, and reorganized based on learning needs, mimicking neurogenesis and synaptic pruning in the developing brain.

```python
class NeurogenesisController:
    """
    Controls dynamic neuron birth, death, and synaptic reorganization.
    
    Inspired by:
    - Neurogenesis: Birth of new neurons (hippocampus, olfactory bulb)
    - Synaptic pruning: Removal of unused connections (especially adolescence)
    - Hebbian rewiring: "Neurons that fire together, wire together"
    - Homeostatic regulation: Maintain network stability
    """
    
    def __init__(self, layer, growth_config):
        self.layer = layer
        self.growth_threshold = growth_config.get('growth_threshold', 0.85)
        self.prune_threshold = growth_config.get('prune_threshold', 0.01)
        self.max_neurons = growth_config.get('max_neurons', 1000)
        self.min_neurons = growth_config.get('min_neurons', 10)
        
        # Track neuron importance and usage
        self.neuron_importance = {}
        self.neuron_age = {}
        self.connection_strength = {}
        
    def should_spawn_neuron(self, layer_stats):
        """
        Decide if new neuron should be created.
        
        Spawn when:
        1. Layer capacity saturated (all neurons active frequently)
        2. High error on specific input patterns
        3. Discovering new concept that needs representation
        4. Developmental signal (phase transition)
        """
        # Capacity check
        capacity = layer_stats['utilization']
        if capacity > self.growth_threshold:
            return True, "capacity_saturated"
        
        # Error check
        if layer_stats['recent_error'] > layer_stats['avg_error'] * 1.5:
            return True, "high_error"
        
        # Concept discovery (from concept hierarchy)
        if layer_stats.get('new_concept_detected', False):
            return True, "new_concept"
        
        # Developmental milestone
        if layer_stats.get('phase_transition', False):
            return True, "developmental"
        
        return False, None
    
    def spawn_neuron(self, reason, parent_neurons=None):
        """
        Create new neuron and integrate into layer.
        
        Strategies:
        1. Random initialization: Start from scratch
        2. Clone from parent: Copy similar neuron + add noise
        3. Interpolate: Average of multiple parent neurons
        4. Specialized: Initialize for specific feature
        """
        # Check capacity
        if len(self.layer.neurons) >= self.max_neurons:
            print(f"Max neurons reached ({self.max_neurons})")
            return None
        
        # Create neuron config (inherit from layer)
        neuron_config = self.layer.base_neuron_config.copy()
        
        if reason == "capacity_saturated":
            # Clone most active neuron + add noise
            most_active = self._find_most_active_neuron()
            new_neuron = self._clone_neuron(most_active, noise_std=0.1)
            
        elif reason == "high_error":
            # Create specialized neuron for hard patterns
            new_neuron = BioInspiredNeuron(
                input_dim=self.layer.input_dim,
                **neuron_config
            )
            # Initialize with larger weights for stronger initial signal
            for param in new_neuron.parameters():
                param.data *= 1.5
                
        elif reason == "new_concept":
            # Specialized for new concept detection
            new_neuron = BioInspiredNeuron(
                input_dim=self.layer.input_dim,
                num_dendrites=8,  # More dendrites for complex concept
                **neuron_config
            )
            
        else:  # developmental or default
            # Standard initialization
            new_neuron = BioInspiredNeuron(
                input_dim=self.layer.input_dim,
                **neuron_config
            )
        
        # Add to layer
        neuron_id = len(self.layer.neurons)
        self.layer.neurons.append(new_neuron)
        
        # Track metadata
        self.neuron_age[neuron_id] = 0
        self.neuron_importance[neuron_id] = 0.5  # Start with medium importance
        
        # Connect to existing neurons (synaptic wiring)
        self._wire_new_neuron(neuron_id, parent_neurons)
        
        print(f"✨ Spawned neuron {neuron_id} (reason: {reason})")
        return new_neuron
    
    def should_prune_neuron(self, neuron_id):
        """
        Decide if neuron should be removed.
        
        Prune when:
        1. Low importance (rarely activated)
        2. Redundant with other neurons (high correlation)
        3. Dead/silent (zero activity for long time)
        4. Developmental cleanup (phase transition)
        """
        importance = self.neuron_importance.get(neuron_id, 0.5)
        age = self.neuron_age.get(neuron_id, 0)
        
        # Don't prune young neurons (give them time to learn)
        if age < 1000:  # Time steps
            return False, None
        
        # Check importance
        if importance < self.prune_threshold:
            return True, "low_importance"
        
        # Check if dead
        neuron = self.layer.neurons[neuron_id]
        if neuron.get_sparsity() < 0.001:  # Active < 0.1% of time
            return True, "dead_neuron"
        
        # Check redundancy
        if self._is_redundant(neuron_id):
            return True, "redundant"
        
        return False, None
    
    def prune_neuron(self, neuron_id, reason):
        """
        Remove neuron from layer and rewire connections.
        """
        if len(self.layer.neurons) <= self.min_neurons:
            print(f"Min neurons reached ({self.min_neurons}), not pruning")
            return False
        
        # Remove neuron
        pruned_neuron = self.layer.neurons.pop(neuron_id)
        
        # Clean up metadata
        del self.neuron_age[neuron_id]
        del self.neuron_importance[neuron_id]
        
        # Rewire connections (redistribute to remaining neurons)
        self._rewire_after_pruning(neuron_id)
        
        print(f"✂️ Pruned neuron {neuron_id} (reason: {reason})")
        return True
    
    def update_importance(self, neuron_id, activity, error_gradient):
        """
        Update neuron importance based on activity and contribution.
        
        Importance factors:
        1. Activity level (how often active)
        2. Error gradient (contribution to learning)
        3. Connection strength (importance to downstream)
        4. Age (newer neurons get benefit of doubt)
        """
        # Exponential moving average
        alpha = 0.01
        
        # Activity contribution
        activity_score = activity / (1.0 + activity)  # Normalize
        
        # Gradient contribution (how much it helps learning)
        gradient_score = abs(error_gradient) / (1.0 + abs(error_gradient))
        
        # Age bonus (newer neurons get higher initial importance)
        age = self.neuron_age.get(neuron_id, 0)
        age_bonus = np.exp(-age / 10000)  # Decays over 10k steps
        
        # Combined importance
        new_importance = (
            0.4 * activity_score +
            0.4 * gradient_score +
            0.2 * age_bonus
        )
        
        # Update with EMA
        old_importance = self.neuron_importance.get(neuron_id, 0.5)
        self.neuron_importance[neuron_id] = (
            (1 - alpha) * old_importance + alpha * new_importance
        )
        
        # Increment age
        self.neuron_age[neuron_id] = age + 1
    
    def reorganize_connections(self):
        """
        Hebbian rewiring: Strengthen frequently co-active connections,
        weaken rarely co-active connections.
        
        "Neurons that fire together, wire together"
        """
        # Track co-activation
        activation_history = self._get_recent_activations(window=100)
        
        # Compute pairwise correlations
        correlations = np.corrcoef(activation_history.T)
        
        for i in range(len(self.layer.neurons)):
            for j in range(i + 1, len(self.layer.neurons)):
                correlation = correlations[i, j]
                
                # Strengthen high-correlation connections
                if correlation > 0.7:
                    self._strengthen_connection(i, j, amount=0.01)
                
                # Weaken low-correlation connections
                elif correlation < 0.1:
                    self._weaken_connection(i, j, amount=0.01)
        
        print(f"🔄 Reorganized connections (Hebbian rewiring)")
    
    def _clone_neuron(self, source_neuron, noise_std=0.1):
        """Clone neuron with small random perturbations"""
        # Create new neuron with same config
        new_neuron = BioInspiredNeuron(
            input_dim=source_neuron.soma_weights.in_features,
            num_dendrites=source_neuron.num_dendrites,
            spiking=source_neuron.spiking
        )
        
        # Copy weights with noise
        with torch.no_grad():
            for new_param, source_param in zip(
                new_neuron.parameters(),
                source_neuron.parameters()
            ):
                new_param.data = source_param.data + torch.randn_like(source_param) * noise_std
        
        return new_neuron
    
    def _wire_new_neuron(self, neuron_id, parent_neurons):
        """Connect new neuron to existing network"""
        if parent_neurons is None:
            # Random connections (small initial weights)
            return
        
        # Connect to parent neurons with higher initial strength
        for parent_id in parent_neurons:
            self.connection_strength[(parent_id, neuron_id)] = 0.5
            self.connection_strength[(neuron_id, parent_id)] = 0.5
    
    def _is_redundant(self, neuron_id):
        """Check if neuron is redundant with others"""
        # Simplified: Check activity correlation
        activation_history = self._get_recent_activations(window=100)
        
        if activation_history.shape[0] < 100:
            return False  # Not enough data
        
        neuron_activity = activation_history[:, neuron_id]
        
        # Check correlation with all other neurons
        for other_id in range(len(self.layer.neurons)):
            if other_id == neuron_id:
                continue
            
            other_activity = activation_history[:, other_id]
            correlation = np.corrcoef(neuron_activity, other_activity)[0, 1]
            
            # If highly correlated, it's redundant
            if correlation > 0.95:
                return True
        
        return False
    
    def _get_recent_activations(self, window=100):
        """Get recent activation history for all neurons"""
        # Placeholder - would track in practice
        return np.random.rand(window, len(self.layer.neurons))
    
    def _find_most_active_neuron(self):
        """Find neuron with highest activity"""
        max_activity = 0
        most_active = None
        
        for neuron in self.layer.neurons:
            activity = neuron.get_sparsity()
            if activity > max_activity:
                max_activity = activity
                most_active = neuron
        
        return most_active
    
    def _strengthen_connection(self, neuron_i, neuron_j, amount):
        """Strengthen synaptic connection between neurons"""
        key = (neuron_i, neuron_j)
        current = self.connection_strength.get(key, 0.1)
        self.connection_strength[key] = min(1.0, current + amount)
    
    def _weaken_connection(self, neuron_i, neuron_j, amount):
        """Weaken synaptic connection between neurons"""
        key = (neuron_i, neuron_j)
        current = self.connection_strength.get(key, 0.1)
        self.connection_strength[key] = max(0.0, current - amount)
    
    def _rewire_after_pruning(self, pruned_id):
        """Redistribute connections after neuron removal"""
        # Remove all connections involving pruned neuron
        to_remove = []
        for key in self.connection_strength.keys():
            if pruned_id in key:
                to_remove.append(key)
        
        for key in to_remove:
            del self.connection_strength[key]
```

**Growth Timeline Across Development:**

```python
# Phase 1 (Months 1-4): Rapid growth
# - Start: 100 neurons per layer
# - Add ~20 new neurons per month
# - Minimal pruning (learning phase)

# Phase 2 (Months 5-8): Continued growth + early pruning
# - Add ~10 new neurons per month
# - Begin pruning dead/redundant neurons
# - Net growth: ~5 neurons per month

# Phase 3 (Months 9-14): Balanced growth and pruning
# - Add ~5 new neurons per month
# - Prune ~3 neurons per month
# - Net growth: ~2 neurons per month
# - Reorganize connections (Hebbian rewiring)

# Phase 4 (Months 15-20): Refinement
# - Add ~2 new neurons per month
# - Prune ~2 neurons per month
# - Net growth: ~0 (stable)
# - Focus on connection optimization

# Phase 5 (Months 21+): Mature, specialized
# - Minimal growth (only for new domains)
# - Strategic pruning for efficiency
# - Heavy connection reorganization
# - Optimize for task-specific efficiency
```

**Integration with Developmental Phases:**

```python
class DevelopmentalNeuronManager:
    """Manage neuron growth according to developmental stage"""
    
    def __init__(self):
        self.phase = 0  # Current developmental phase
        self.neurogenesis_controllers = {}  # Per-layer controllers
        
    def set_phase(self, phase):
        """Update developmental phase and adjust growth parameters"""
        self.phase = phase
        
        if phase == 1:
            # Infancy: Rapid neuron proliferation
            self._configure_rapid_growth()
        elif phase == 2:
            # Toddler: Continued growth, early pruning
            self._configure_balanced_growth()
        elif phase == 3:
            # Child: Selective growth, active pruning
            self._configure_selective_growth()
        elif phase == 4:
            # Adolescent: Refinement, heavy pruning
            self._configure_refinement()
        elif phase == 5:
            # Adult: Stability, minimal changes
            self._configure_mature()
    
    def _configure_rapid_growth(self):
        """Phase 1: Rapid proliferation"""
        for controller in self.neurogenesis_controllers.values():
            controller.growth_threshold = 0.7  # Lower threshold (grow easily)
            controller.prune_threshold = 0.001  # High threshold (prune rarely)
    
    def _configure_balanced_growth(self):
        """Phase 2: Balanced"""
        for controller in self.neurogenesis_controllers.values():
            controller.growth_threshold = 0.8
            controller.prune_threshold = 0.01
    
    def _configure_selective_growth(self):
        """Phase 3: Selective"""
        for controller in self.neurogenesis_controllers.values():
            controller.growth_threshold = 0.85
            controller.prune_threshold = 0.02
    
    def _configure_refinement(self):
        """Phase 4: Refinement (adolescent pruning)"""
        for controller in self.neurogenesis_controllers.values():
            controller.growth_threshold = 0.9  # Higher threshold (grow rarely)
            controller.prune_threshold = 0.05  # Lower threshold (prune aggressively)
    
    def _configure_mature(self):
        """Phase 5: Mature stability"""
        for controller in self.neurogenesis_controllers.values():
            controller.growth_threshold = 0.95  # Very high (almost no growth)
            controller.prune_threshold = 0.1  # Aggressive pruning
```

This neurogenesis system enables E-Brain to dynamically adapt its structure based on learning needs, mimicking the brain's developmental trajectory from rapid growth in infancy to selective refinement in adulthood.
