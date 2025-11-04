# E-Brain Technical Architecture

## Overview

E-Brain is designed as a modular, dynamically growing neural system that mimics developmental stages of human learning. The architecture emphasizes flexibility, growth capability, and multi-modal integration.

## 0. Foundational Component: Bio-Inspired Neuron

Before diving into system architecture, we define the fundamental building block: an enhanced artificial neuron that mimics key biological properties.

### 0.1 Biological Neurons vs Traditional Artificial Neurons

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

### 0.2 Enhanced Bio-Inspired Neuron Design

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

### 0.3 Key Features Explained

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

### 0.4 Usage in E-Brain Architecture

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

### 0.5 Comparison Table

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

### 0.6 Advantages for E-Brain

1. **Temporal Learning**: Better at sequences, time series, temporal credit assignment
2. **Local Plasticity**: Can learn online during inference, not just training
3. **Energy Efficiency**: Sparse activation reduces computation (like real brain)
4. **Biological Plausibility**: More interpretable, closer to neuroscience
5. **Neuromodulation**: Natural integration with reward system (dopamine-like signals)
6. **Homeostasis**: Self-regulating, prevents overfitting or dead neurons
7. **Developmental Growth**: Easier to add/remove neurons dynamically

### 0.7 Implementation Strategy

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

### 0.8 Neuron Growth, Pruning, and Reorganization

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

#### Concurrent Thought Processing System

E-Brain can maintain and switch between multiple concurrent "thoughts" or reasoning threads, mimicking the brain's ability to work on multiple problems simultaneously.

```python
class ConcurrentThoughtSystem:
    """
    Manages multiple parallel thought streams (reasoning threads).
    
    Inspired by human cognitive capabilities:
    - Working memory slots (4-7 concurrent thoughts)
    - Rapid context switching (100-200ms in humans)
    - Background processing (subconscious thoughts)
    - Thought persistence (suspend/resume)
    - Cross-pollination (insights from one thought help another)
    
    Examples:
    - Solve math problem while reading text
    - Multiple hypothesis exploration in parallel
    - Background: "cook" difficult problems while working on easier ones
    - Creative: combine insights from different thought streams
    """
    
    def __init__(
        self,
        max_concurrent_thoughts=7,  # Working memory limit
        context_switch_cost=0.1,  # Time/compute cost to switch
        background_slots=3  # How many can run in background
    ):
        self.max_concurrent_thoughts = max_concurrent_thoughts
        self.context_switch_cost = context_switch_cost
        self.background_slots = background_slots
        
        # Active thought streams
        self.thoughts = {}  # thought_id -> ThoughtStream
        self.focused_thought = None  # Currently active thought
        self.background_thoughts = []  # Thoughts running in background
        
        # Attention controller
        self.attention_controller = AttentionController()
        
        # Shared memory for cross-pollination
        self.shared_insight_memory = SharedInsightMemory()
        
        # Performance tracking
        self.thought_creation_count = 0
        self.context_switches = 0
        
    def create_thought(self, task, initial_state, priority="medium"):
        """
        Spawn a new thought stream for a task.
        
        Args:
            task: Task description/goal
            initial_state: Starting state for reasoning
            priority: "high", "medium", "low"
            
        Returns:
            thought_id: Unique identifier for this thought
        """
        # Check capacity
        if len(self.thoughts) >= self.max_concurrent_thoughts:
            # Need to suspend lowest priority thought
            self._suspend_lowest_priority_thought()
        
        # Create new thought stream
        thought_id = f"thought_{self.thought_creation_count}"
        self.thought_creation_count += 1
        
        thought = ThoughtStream(
            thought_id=thought_id,
            task=task,
            initial_state=initial_state,
            priority=priority,
            shared_memory=self.shared_insight_memory
        )
        
        self.thoughts[thought_id] = thought
        
        # Decide placement: focused, background, or suspended
        if self.focused_thought is None:
            self.focused_thought = thought_id
            thought.set_mode("focused")
        elif len(self.background_thoughts) < self.background_slots:
            self.background_thoughts.append(thought_id)
            thought.set_mode("background")
        else:
            thought.set_mode("suspended")
        
        return thought_id
    
    def step(self, timestep):
        """
        Execute one step of concurrent thought processing.
        
        - Focused thought gets most compute
        - Background thoughts get minimal compute (keep alive)
        - Suspended thoughts inactive
        - Periodically check if attention should switch
        """
        results = {}
        
        # 1. Process focused thought (full compute)
        if self.focused_thought:
            focused = self.thoughts[self.focused_thought]
            result = focused.step(compute_budget=1.0)
            results[self.focused_thought] = result
            
            # Check if focused thought completed
            if focused.is_completed():
                self._handle_completion(self.focused_thought)
        
        # 2. Process background thoughts (limited compute)
        for thought_id in self.background_thoughts[:]:
            thought = self.thoughts.get(thought_id)
            if thought:
                # Background gets 10% compute
                result = thought.step(compute_budget=0.1)
                results[thought_id] = result
                
                # Check completion
                if thought.is_completed():
                    self._handle_completion(thought_id)
        
        # 3. Decide if attention should switch
        if timestep % 10 == 0:  # Check every 10 steps
            should_switch, new_focus = self.attention_controller.should_switch_attention(
                thoughts=self.thoughts,
                current_focus=self.focused_thought
            )
            
            if should_switch and new_focus != self.focused_thought:
                self._switch_attention(new_focus)
        
        # 4. Cross-pollinate insights
        if timestep % 50 == 0:  # Periodically
            self._cross_pollinate_insights()
        
        return results
    
    def _switch_attention(self, new_focus_id):
        """
        Switch attention from current focus to new thought.
        
        Context switching:
        - Save current thought state
        - Load new thought state
        - Adjust priority queues
        """
        if self.focused_thought == new_focus_id:
            return  # Already focused
        
        # Save current focus to background
        if self.focused_thought:
            old_thought = self.thoughts[self.focused_thought]
            old_thought.set_mode("background")
            self.background_thoughts.append(self.focused_thought)
        
        # Promote new focus
        new_thought = self.thoughts[new_focus_id]
        new_thought.set_mode("focused")
        
        # Remove from background if present
        if new_focus_id in self.background_thoughts:
            self.background_thoughts.remove(new_focus_id)
        
        self.focused_thought = new_focus_id
        self.context_switches += 1
        
        # Context switch cost (simulate)
        time.sleep(self.context_switch_cost)
    
    def _cross_pollinate_insights(self):
        """
        Share insights between thought streams.
        
        Examples:
        - Thought A discovers pattern → share with Thought B
        - Thought B solving similar problem → transfer strategy
        - Creative combination of ideas from multiple streams
        """
        # Collect insights from all active thoughts
        insights = []
        for thought_id, thought in self.thoughts.items():
            if thought.mode != "suspended":
                thought_insights = thought.extract_insights()
                insights.extend(thought_insights)
        
        # Store in shared memory
        for insight in insights:
            self.shared_insight_memory.store(insight)
        
        # Distribute relevant insights to other thoughts
        for thought_id, thought in self.thoughts.items():
            relevant_insights = self.shared_insight_memory.retrieve_relevant(
                thought.task,
                k=5
            )
            thought.receive_insights(relevant_insights)
    
    def _handle_completion(self, thought_id):
        """Handle thought stream completion"""
        thought = self.thoughts[thought_id]
        
        # Extract final solution and insights
        solution = thought.get_solution()
        insights = thought.extract_insights()
        
        # Store insights for future thoughts
        for insight in insights:
            self.shared_insight_memory.store(insight)
        
        # Remove from active tracking
        if self.focused_thought == thought_id:
            self.focused_thought = None
        if thought_id in self.background_thoughts:
            self.background_thoughts.remove(thought_id)
        
        # Mark as completed (keep in memory for a while)
        thought.set_mode("completed")
        
        return solution
    
    def _suspend_lowest_priority_thought(self):
        """Suspend lowest priority thought to make room"""
        # Find lowest priority non-focused thought
        lowest_priority = None
        lowest_score = float('inf')
        
        for thought_id, thought in self.thoughts.items():
            if thought_id != self.focused_thought:
                priority_score = thought.get_priority_score()
                if priority_score < lowest_score:
                    lowest_score = priority_score
                    lowest_priority = thought_id
        
        if lowest_priority:
            thought = self.thoughts[lowest_priority]
            thought.set_mode("suspended")
            
            if lowest_priority in self.background_thoughts:
                self.background_thoughts.remove(lowest_priority)
    
    def get_status(self):
        """Get current status of all thoughts"""
        return {
            'focused': self.focused_thought,
            'background': self.background_thoughts,
            'total_thoughts': len(self.thoughts),
            'context_switches': self.context_switches,
            'thoughts': {
                tid: {
                    'mode': t.mode,
                    'progress': t.get_progress(),
                    'priority': t.priority
                }
                for tid, t in self.thoughts.items()
            }
        }


class ThoughtStream:
    """
    Individual thought stream (reasoning thread).
    
    Maintains:
    - Task context
    - Reasoning state
    - Partial solutions
    - Insights discovered
    - Computational budget used
    """
    
    def __init__(self, thought_id, task, initial_state, priority, shared_memory):
        self.thought_id = thought_id
        self.task = task
        self.state = initial_state
        self.priority = priority
        self.shared_memory = shared_memory
        
        # Reasoning state
        self.reasoning_trace = []
        self.partial_solutions = []
        self.insights = []
        
        # Mode: "focused", "background", "suspended", "completed"
        self.mode = "suspended"
        
        # Progress tracking
        self.steps_taken = 0
        self.compute_used = 0.0
        self.start_time = time.time()
        
        # Stage processor (for multi-stage reasoning)
        self.stage_processor = StageProcessor(depth=0)
        self.current_stage = 0
        
    def step(self, compute_budget):
        """
        Execute one reasoning step.
        
        Args:
            compute_budget: 0.0-1.0, fraction of full compute available
        """
        if self.mode == "suspended":
            return None
        
        # Adjust reasoning depth based on compute budget
        if compute_budget >= 0.8:
            # Full reasoning
            num_stages = 3
        elif compute_budget >= 0.3:
            # Medium reasoning
            num_stages = 2
        else:
            # Minimal reasoning (keep alive)
            num_stages = 1
        
        # Perform reasoning step
        for _ in range(num_stages):
            # Check shared memory for relevant insights
            relevant_insights = self.shared_memory.retrieve_relevant(
                self.task,
                k=3
            )
            
            # Incorporate insights into reasoning
            state_with_insights = self._incorporate_insights(
                self.state,
                relevant_insights
            )
            
            # Process current stage
            stage_output = self.stage_processor(
                state_with_insights,
                context=self.reasoning_trace
            )
            
            # Update state
            self.state = stage_output.next_state
            
            # Record reasoning step
            self.reasoning_trace.append({
                'step': self.steps_taken,
                'stage': self.current_stage,
                'output': stage_output.prediction,
                'confidence': stage_output.confidence
            })
            
            # Extract insights
            if stage_output.insight:
                self.insights.append(stage_output.insight)
            
            # Check if solution found
            if stage_output.solution_found:
                self.partial_solutions.append(stage_output.solution)
            
            self.current_stage += 1
        
        self.steps_taken += 1
        self.compute_used += compute_budget
        
        return {
            'state': self.state,
            'confidence': stage_output.confidence,
            'insights': len(self.insights)
        }
    
    def set_mode(self, mode):
        """Set thought mode: focused, background, suspended, completed"""
        self.mode = mode
    
    def is_completed(self):
        """Check if thought has reached solution"""
        if not self.partial_solutions:
            return False
        
        # Check if latest solution has high confidence
        latest_trace = self.reasoning_trace[-1] if self.reasoning_trace else None
        if latest_trace and latest_trace['confidence'] > 0.9:
            return True
        
        # Or if we've exhausted reasonable compute
        if self.steps_taken > 1000:
            return True
        
        return False
    
    def extract_insights(self):
        """Extract insights discovered during reasoning"""
        return self.insights
    
    def get_solution(self):
        """Get final solution"""
        if self.partial_solutions:
            return self.partial_solutions[-1]
        return None
    
    def receive_insights(self, insights):
        """Receive insights from other thought streams"""
        # Incorporate external insights
        for insight in insights:
            if self._is_relevant_insight(insight):
                self.insights.append({
                    'insight': insight,
                    'source': 'cross_pollination'
                })
    
    def _incorporate_insights(self, state, insights):
        """Incorporate insights into current reasoning state"""
        if not insights:
            return state
        
        # Augment state with insight information
        state_augmented = state.copy()
        state_augmented['external_insights'] = insights
        return state_augmented
    
    def _is_relevant_insight(self, insight):
        """Check if insight relevant to current task"""
        # Simplified: check keyword overlap
        task_keywords = set(self.task.lower().split())
        insight_keywords = set(str(insight).lower().split())
        overlap = task_keywords.intersection(insight_keywords)
        return len(overlap) > 0
    
    def get_progress(self):
        """Get progress estimate (0.0-1.0)"""
        if not self.reasoning_trace:
            return 0.0
        
        # Average confidence over recent steps
        recent = self.reasoning_trace[-10:]
        avg_confidence = np.mean([r['confidence'] for r in recent])
        return avg_confidence
    
    def get_priority_score(self):
        """Calculate priority score for attention allocation"""
        priority_values = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
        base_priority = priority_values.get(self.priority, 1.0)
        
        # Boost priority if making progress
        progress = self.get_progress()
        
        # Boost priority if urgent (time-sensitive)
        time_elapsed = time.time() - self.start_time
        urgency_factor = min(time_elapsed / 60.0, 2.0)  # Cap at 2x
        
        return base_priority * (1.0 + progress) * urgency_factor


class AttentionController:
    """
    Decides which thought should receive focused attention.
    
    Factors:
    - Priority (user-defined or task-based)
    - Progress (making headway vs stuck)
    - Urgency (time-sensitive tasks)
    - Deadlock (background thought needs focus to proceed)
    """
    
    def should_switch_attention(self, thoughts, current_focus):
        """
        Decide if attention should switch to different thought.
        
        Returns:
            (should_switch: bool, new_focus_id: str)
        """
        if current_focus is None:
            # No current focus, pick highest priority
            return True, self._select_highest_priority(thoughts)
        
        current_thought = thoughts.get(current_focus)
        if not current_thought:
            return True, self._select_highest_priority(thoughts)
        
        # Check if current thought is stuck
        if self._is_stuck(current_thought):
            # Switch to different thought, let current "cook" in background
            return True, self._select_alternate(thoughts, exclude=current_focus)
        
        # Check if any thought has much higher priority
        highest_priority_id = self._select_highest_priority(thoughts)
        if highest_priority_id != current_focus:
            highest_thought = thoughts[highest_priority_id]
            current_priority = current_thought.get_priority_score()
            highest_priority = highest_thought.get_priority_score()
            
            # Switch if priority difference significant
            if highest_priority > current_priority * 1.5:
                return True, highest_priority_id
        
        # Stay focused on current thought
        return False, current_focus
    
    def _is_stuck(self, thought):
        """Check if thought is making progress"""
        if len(thought.reasoning_trace) < 10:
            return False  # Too early to tell
        
        # Check if confidence plateaued
        recent = thought.reasoning_trace[-10:]
        confidences = [r['confidence'] for r in recent]
        confidence_variance = np.var(confidences)
        
        # Low variance = stuck
        return confidence_variance < 0.01
    
    def _select_highest_priority(self, thoughts):
        """Select thought with highest priority score"""
        best_id = None
        best_score = -1
        
        for thought_id, thought in thoughts.items():
            if thought.mode != "suspended":
                score = thought.get_priority_score()
                if score > best_score:
                    best_score = score
                    best_id = thought_id
        
        return best_id
    
    def _select_alternate(self, thoughts, exclude):
        """Select alternate thought (not current focus)"""
        best_id = None
        best_score = -1
        
        for thought_id, thought in thoughts.items():
            if thought_id != exclude and thought.mode != "suspended":
                score = thought.get_priority_score()
                if score > best_score:
                    best_score = score
                    best_id = thought_id
        
        return best_id if best_id else exclude


class SharedInsightMemory:
    """
    Shared memory for insights across thought streams.
    Enables cross-pollination of ideas.
    """
    
    def __init__(self, capacity=1000):
        self.insights = []
        self.capacity = capacity
        self.insight_embeddings = []
        
    def store(self, insight):
        """Store insight with embedding"""
        if len(self.insights) >= self.capacity:
            # Remove oldest
            self.insights.pop(0)
            self.insight_embeddings.pop(0)
        
        # Generate embedding (simplified)
        embedding = self._embed_insight(insight)
        
        self.insights.append(insight)
        self.insight_embeddings.append(embedding)
    
    def retrieve_relevant(self, query, k=5):
        """Retrieve k most relevant insights for query"""
        if not self.insights:
            return []
        
        query_embedding = self._embed_insight(query)
        
        # Compute similarities
        similarities = []
        for i, emb in enumerate(self.insight_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((sim, i))
        
        # Sort and get top k
        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]
        
        return [self.insights[i] for i in top_k_indices]
    
    def _embed_insight(self, insight):
        """Generate embedding for insight (simplified)"""
        # In practice, use model's embedding layer
        return torch.randn(512)  # Placeholder
    
    def _cosine_similarity(self, a, b):
        """Compute cosine similarity"""
        return torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
```

**Usage Example:**

```python
# Initialize concurrent thought system
thought_system = ConcurrentThoughtSystem(
    max_concurrent_thoughts=7,  # Working memory limit
    background_slots=3
)

# Spawn multiple thought streams
math_thought = thought_system.create_thought(
    task="Solve differential equation",
    initial_state=problem_state,
    priority="high"
)

text_thought = thought_system.create_thought(
    task="Summarize article",
    initial_state=text_state,
    priority="medium"
)

code_thought = thought_system.create_thought(
    task="Debug code",
    initial_state=code_state,
    priority="low"
)

# Run concurrent processing
for timestep in range(1000):
    results = thought_system.step(timestep)
    
    # Thoughts work in parallel:
    # - Focused thought gets 100% compute
    # - Background thoughts get 10% compute (keep alive)
    # - System automatically switches attention when stuck or priorities change
    # - Insights from one thought help others

# Get status
status = thought_system.get_status()
print(f"Focused: {status['focused']}")
print(f"Background: {status['background']}")
print(f"Context switches: {status['context_switches']}")
```

**Benefits:**

1. **Parallel Problem Solving**: Work on multiple problems simultaneously
2. **Background Processing**: Difficult problems "cook" in background while working on easier ones
3. **Context Switching**: Rapid switching between thoughts (like human attention)
4. **Cross-Pollination**: Insights from one problem help solve another (creativity!)
5. **Efficiency**: Don't waste compute waiting for one slow thought
6. **Resilience**: If stuck on one problem, switch to another
7. **Time Management**: Urgent tasks can preempt less important ones

**Developmental Progression:**

- **Phase 1-2**: Single thought (no concurrency yet)
- **Phase 3**: 2-3 concurrent thoughts (basic multitasking)
- **Phase 4**: 5-7 concurrent thoughts (full working memory)
- **Phase 5**: Advanced attention switching and cross-pollination

#### Internal Timing and Clock System

E-Brain maintains multiple hierarchical timing mechanisms, mimicking the brain's ability to track time at different scales, predict temporal patterns, and time actions accurately.

```python
class InternalTimingSystem:
    """
    Multi-scale timing system inspired by brain's temporal processing.
    
    Biological inspiration:
    - Circadian rhythms (24-hour cycle) → developmental cycles
    - Interval timing (seconds to minutes) → task duration, action timing
    - Millisecond timing (motor control, perception) → precise synchronization
    - Predictive timing (anticipate events) → temporal prediction
    - Sleep cycles (consolidation) → rest periods for memory optimization
    
    Key capabilities:
    - Track elapsed time at multiple scales
    - Predict 'when' events will occur
    - Time actions accurately based on context
    - Schedule sleep/consolidation cycles
    - Learn temporal patterns and rhythms
    """
    
    def __init__(
        self,
        base_tick_ms=10,  # Base clock tick (10ms = 100Hz)
        enable_circadian=True,
        enable_sleep_cycles=True
    ):
        # Multi-scale clocks
        self.base_tick_ms = base_tick_ms
        self.millisecond_timer = MillisecondTimer(tick_ms=base_tick_ms)
        self.interval_timer = IntervalTimer()  # Seconds to minutes
        self.circadian_clock = CircadianClock() if enable_circadian else None
        self.developmental_timer = DevelopmentalTimer()  # Tracks phases
        
        # Temporal prediction
        self.temporal_predictor = TemporalPredictor()
        
        # Action timing
        self.action_scheduler = ActionScheduler()
        
        # Sleep/consolidation system
        self.sleep_system = SleepConsolidationSystem() if enable_sleep_cycles else None
        
        # Performance tracking
        self.timing_accuracy_history = []
        
    def tick(self):
        """
        Advance all clocks by one base tick.
        Called every forward pass or at regular intervals.
        """
        # Advance base timer
        self.millisecond_timer.tick()
        
        # Update interval timer
        if self.millisecond_timer.ticks % 100 == 0:  # Every second
            self.interval_timer.tick()
        
        # Update circadian clock
        if self.circadian_clock and self.interval_timer.seconds % 3600 == 0:  # Every hour
            self.circadian_clock.tick()
        
        # Check if sleep cycle needed
        if self.sleep_system and self.should_sleep():
            self.enter_sleep_cycle()
    
    def get_current_time(self, scale='millisecond'):
        """
        Get current time at specified scale.
        
        Args:
            scale: 'millisecond', 'second', 'minute', 'hour', 'day', 'phase'
        """
        if scale == 'millisecond':
            return self.millisecond_timer.elapsed_ms
        elif scale == 'second':
            return self.interval_timer.seconds
        elif scale == 'minute':
            return self.interval_timer.minutes
        elif scale == 'hour':
            return self.circadian_clock.hours if self.circadian_clock else 0
        elif scale == 'day':
            return self.circadian_clock.days if self.circadian_clock else 0
        elif scale == 'phase':
            return self.developmental_timer.current_phase
    
    def predict_event_time(self, event_type, context):
        """
        Predict when an event will occur based on learned patterns.
        
        Examples:
        - "User typically responds after 2 seconds"
        - "Training epoch completes in 5 minutes"
        - "Reward signal arrives 500ms after action"
        """
        return self.temporal_predictor.predict(event_type, context)
    
    def schedule_action(self, action, target_time_ms, context):
        """
        Schedule action to execute at precise time.
        
        Args:
            action: Action to execute
            target_time_ms: Target execution time in milliseconds
            context: Situational context for timing adjustment
        """
        current_time = self.millisecond_timer.elapsed_ms
        delay = target_time_ms - current_time
        
        # Learn optimal timing from context
        adjusted_delay = self.action_scheduler.adjust_timing(
            action, delay, context
        )
        
        self.action_scheduler.schedule(action, adjusted_delay)
    
    def learn_temporal_pattern(self, event_sequence):
        """
        Learn temporal patterns from event sequences.
        
        Examples:
        - User interaction patterns (morning: questions, evening: coding)
        - Task rhythms (fast at start, slower when tired)
        - Environmental cycles (dataset availability, compute resources)
        """
        self.temporal_predictor.learn_pattern(event_sequence)
    
    def should_sleep(self):
        """
        Determine if E-Brain should enter sleep/consolidation cycle.
        
        Triggers:
        - Fixed schedule (e.g., every 4 hours of active time)
        - After intensive learning (high cognitive load)
        - Low priority periods (no urgent tasks)
        - Memory buffer saturation (needs consolidation)
        """
        if not self.sleep_system:
            return False
        
        active_time = self.interval_timer.get_active_time()
        cognitive_load = self.estimate_cognitive_load()
        memory_pressure = self.get_memory_buffer_usage()
        
        return (
            active_time > self.sleep_system.sleep_interval or
            cognitive_load > 0.8 or
            memory_pressure > 0.9
        )
    
    def enter_sleep_cycle(self):
        """
        Enter sleep/consolidation mode.
        
        During sleep:
        - Memory consolidation (episodic → semantic)
        - Experience replay for reinforcement learning
        - Concept refinement and pruning
        - Connection reweighting
        - Homeostatic scaling
        """
        print(f"[SLEEP] Entering consolidation cycle at t={self.get_current_time('second')}s")
        
        # Pause active processing
        self.pause_active_thoughts()
        
        # Run consolidation
        self.sleep_system.consolidate(
            episodic_memory=self.episodic_buffer,
            semantic_memory=self.semantic_memory,
            concept_graph=self.concept_graph
        )
        
        # Optimize neural weights
        self.sleep_system.optimize_connections(self.network)
        
        # Resume
        self.resume_active_thoughts()
        
        print(f"[SLEEP] Consolidation complete. Duration: {self.sleep_system.last_duration}s")
    
    def estimate_cognitive_load(self):
        """Estimate current cognitive/computational load (0.0-1.0)"""
        # Based on: active thoughts, processing speed, error rates
        return 0.5  # Placeholder
    
    def get_memory_buffer_usage(self):
        """Get episodic memory buffer usage (0.0-1.0)"""
        return 0.3  # Placeholder


class MillisecondTimer:
    """High-precision timer for millisecond-scale timing"""
    
    def __init__(self, tick_ms=10):
        self.tick_ms = tick_ms
        self.ticks = 0
        self.elapsed_ms = 0
        self.real_start_time = time.time()
    
    def tick(self):
        """Advance timer by one tick"""
        self.ticks += 1
        self.elapsed_ms += self.tick_ms
    
    def reset(self):
        """Reset timer"""
        self.ticks = 0
        self.elapsed_ms = 0
        self.real_start_time = time.time()
    
    def sync_with_real_time(self):
        """Synchronize with wall clock time"""
        real_elapsed = (time.time() - self.real_start_time) * 1000
        drift = real_elapsed - self.elapsed_ms
        return drift


class IntervalTimer:
    """Seconds-to-minutes scale timing"""
    
    def __init__(self):
        self.seconds = 0
        self.minutes = 0
        self.hours = 0
        self.active_time = 0  # Exclude sleep time
        self.is_active = True
    
    def tick(self):
        """Advance by one second"""
        self.seconds += 1
        if self.is_active:
            self.active_time += 1
        
        if self.seconds >= 60:
            self.seconds = 0
            self.minutes += 1
        
        if self.minutes >= 60:
            self.minutes = 0
            self.hours += 1
    
    def get_active_time(self):
        """Get active time in seconds (excluding sleep)"""
        return self.active_time
    
    def pause(self):
        """Pause active time tracking (during sleep)"""
        self.is_active = False
    
    def resume(self):
        """Resume active time tracking"""
        self.is_active = True


class CircadianClock:
    """24-hour cycle simulation for developmental rhythms"""
    
    def __init__(self, cycle_length_hours=24):
        self.cycle_length = cycle_length_hours
        self.hours = 0
        self.days = 0
        self.phase = 0.0  # 0.0-1.0 within cycle
    
    def tick(self):
        """Advance by one hour"""
        self.hours += 1
        
        if self.hours >= self.cycle_length:
            self.hours = 0
            self.days += 1
        
        self.phase = self.hours / self.cycle_length
    
    def get_phase(self):
        """Get current phase in cycle (0.0-1.0)"""
        return self.phase
    
    def is_active_period(self):
        """Check if in active period (vs rest period)"""
        # Active: 6am-10pm (0.25-0.917 of cycle)
        return 0.25 <= self.phase <= 0.917


class DevelopmentalTimer:
    """Tracks developmental phases and transitions"""
    
    def __init__(self):
        self.current_phase = 1
        self.phase_start_time = 0
        self.phase_durations = {
            1: 3 * 30 * 24 * 3600,   # 3 months in seconds
            2: 3 * 30 * 24 * 3600,   # 3 months
            3: 6 * 30 * 24 * 3600,   # 6 months
            4: 6 * 30 * 24 * 3600,   # 6 months
            5: float('inf')           # Ongoing
        }
    
    def check_phase_transition(self, elapsed_seconds):
        """Check if should transition to next phase"""
        time_in_phase = elapsed_seconds - self.phase_start_time
        
        if time_in_phase >= self.phase_durations[self.current_phase]:
            self.transition_to_next_phase(elapsed_seconds)
    
    def transition_to_next_phase(self, elapsed_seconds):
        """Transition to next developmental phase"""
        if self.current_phase < 5:
            self.current_phase += 1
            self.phase_start_time = elapsed_seconds
            print(f"[DEVELOPMENT] Transitioned to Phase {self.current_phase}")


class TemporalPredictor:
    """
    Learns and predicts temporal patterns.
    
    Examples:
    - "User responds after ~2 seconds"
    - "Reward arrives 500ms after action"
    - "Task typically takes 5 minutes"
    - "Data available every 10 seconds"
    """
    
    def __init__(self):
        self.event_patterns = {}  # event_type -> temporal distribution
        self.sequence_patterns = []  # Learned event sequences with timing
        
        # Temporal encoding (like positional encoding)
        self.temporal_encoder = TemporalEncoder()
    
    def predict(self, event_type, context):
        """
        Predict when event will occur.
        
        Returns:
            expected_delay_ms: Expected time until event
            confidence: Prediction confidence
        """
        if event_type not in self.event_patterns:
            return None, 0.0
        
        pattern = self.event_patterns[event_type]
        
        # Context-dependent adjustment
        base_delay = pattern['mean_delay_ms']
        adjusted_delay = self.adjust_for_context(base_delay, context, pattern)
        
        confidence = pattern['confidence']
        
        return adjusted_delay, confidence
    
    def learn_pattern(self, event_sequence):
        """
        Learn temporal pattern from event sequence.
        
        Args:
            event_sequence: [(event_type, timestamp), ...]
        """
        # Extract inter-event intervals
        for i in range(len(event_sequence) - 1):
            current_event, current_time = event_sequence[i]
            next_event, next_time = event_sequence[i + 1]
            
            interval = next_time - current_time
            
            # Update statistics for this event transition
            key = (current_event, next_event)
            if key not in self.event_patterns:
                self.event_patterns[key] = {
                    'mean_delay_ms': interval,
                    'variance': 0.0,
                    'count': 1,
                    'confidence': 0.3
                }
            else:
                pattern = self.event_patterns[key]
                # Update running statistics
                old_mean = pattern['mean_delay_ms']
                pattern['count'] += 1
                pattern['mean_delay_ms'] = (old_mean * (pattern['count'] - 1) + interval) / pattern['count']
                pattern['variance'] = 0.9 * pattern['variance'] + 0.1 * (interval - old_mean) ** 2
                pattern['confidence'] = min(0.95, 0.3 + 0.65 * (pattern['count'] / 100))
    
    def adjust_for_context(self, base_delay, context, pattern):
        """Adjust predicted delay based on context"""
        # Context factors: time of day, cognitive load, task difficulty
        adjustment = 1.0
        
        if 'time_of_day' in context:
            # Slower in evening (fatigue)
            if context['time_of_day'] > 0.7:
                adjustment *= 1.2
        
        if 'cognitive_load' in context:
            # Slower under high load
            adjustment *= (1.0 + 0.5 * context['cognitive_load'])
        
        return base_delay * adjustment


class ActionScheduler:
    """
    Schedules and times actions precisely.
    
    Learns optimal timing from experience.
    """
    
    def __init__(self):
        self.scheduled_actions = []  # (action, execute_time_ms)
        self.timing_history = []  # (action, target_time, actual_time, success)
    
    def schedule(self, action, delay_ms):
        """Schedule action for future execution"""
        execute_time = time.time() * 1000 + delay_ms
        self.scheduled_actions.append((action, execute_time))
        self.scheduled_actions.sort(key=lambda x: x[1])  # Sort by time
    
    def check_ready_actions(self, current_time_ms):
        """Get actions ready for execution"""
        ready = []
        remaining = []
        
        for action, execute_time in self.scheduled_actions:
            if current_time_ms >= execute_time:
                ready.append(action)
            else:
                remaining.append((action, execute_time))
        
        self.scheduled_actions = remaining
        return ready
    
    def adjust_timing(self, action, delay_ms, context):
        """
        Adjust action timing based on learned patterns.
        
        Example: Learn to respond slightly faster/slower based on context
        """
        # Look up historical timing for similar actions
        similar_history = [
            h for h in self.timing_history
            if h[0] == action and h[3]  # Same action, successful
        ]
        
        if len(similar_history) > 5:
            # Calculate average adjustment
            adjustments = [h[2] - h[1] for h in similar_history[-10:]]
            avg_adjustment = np.mean(adjustments)
            
            # Apply learned adjustment
            return delay_ms + avg_adjustment
        
        return delay_ms
    
    def record_timing(self, action, target_time, actual_time, success):
        """Record action timing for learning"""
        self.timing_history.append((action, target_time, actual_time, success))
        
        # Keep limited history
        if len(self.timing_history) > 1000:
            self.timing_history = self.timing_history[-1000:]


class SleepConsolidationSystem:
    """
    Simulates sleep cycles for memory consolidation and optimization.
    
    Biological inspiration:
    - Sleep consolidates memories (hippocampus → cortex)
    - REM sleep: creative connections, random replay
    - Deep sleep: synaptic scaling, pruning
    - Circadian rhythm: regular sleep schedule
    """
    
    def __init__(self, sleep_interval_seconds=4*3600):
        self.sleep_interval = sleep_interval_seconds  # 4 hours default
        self.last_sleep_time = 0
        self.last_duration = 0
        self.total_sleep_time = 0
    
    def consolidate(self, episodic_memory, semantic_memory, concept_graph):
        """
        Consolidate memories during sleep.
        
        Process:
        1. Replay episodic memories (experience replay)
        2. Extract patterns and transfer to semantic memory
        3. Strengthen important connections
        4. Prune weak/redundant connections
        5. Refine concepts in concept graph
        """
        start_time = time.time()
        
        # 1. Experience Replay (like REM sleep)
        print("[SLEEP] Experience replay...")
        important_episodes = self.select_important_episodes(episodic_memory)
        for episode in important_episodes:
            # Replay with slight noise (creative recombination)
            self.replay_episode(episode, noise=0.1)
        
        # 2. Memory Transfer (episodic → semantic)
        print("[SLEEP] Memory consolidation...")
        patterns = self.extract_patterns(important_episodes)
        for pattern in patterns:
            semantic_memory.store(pattern)
        
        # 3. Concept Refinement
        print("[SLEEP] Concept refinement...")
        self.refine_concepts(concept_graph)
        
        # 4. Synaptic Scaling (homeostatic plasticity)
        print("[SLEEP] Synaptic scaling...")
        self.homeostatic_scaling()
        
        self.last_duration = time.time() - start_time
        self.total_sleep_time += self.last_duration
    
    def optimize_connections(self, network):
        """
        Optimize neural connections during sleep.
        
        - Strengthen frequently used connections
        - Weaken rarely used connections
        - Prune near-zero weights
        - Renormalize to prevent drift
        """
        print("[SLEEP] Optimizing connections...")
        
        with torch.no_grad():
            for module in network.modules():
                if hasattr(module, 'weight'):
                    # Prune small weights
                    mask = torch.abs(module.weight) > 0.01
                    module.weight *= mask.float()
                    
                    # Renormalize
                    module.weight /= (torch.norm(module.weight) + 1e-8)
    
    def select_important_episodes(self, episodic_memory):
        """Select important episodes for replay"""
        # Prioritize: high reward, surprising, recent
        return episodic_memory.get_prioritized_samples(k=100)
    
    def replay_episode(self, episode, noise=0.0):
        """Replay episode (possibly with noise for creativity)"""
        # Simplified: in practice, run episode through network
        pass
    
    def extract_patterns(self, episodes):
        """Extract common patterns from episodes"""
        patterns = []
        # Simplified: cluster similar episodes, extract prototypes
        return patterns
    
    def refine_concepts(self, concept_graph):
        """Refine and prune concepts"""
        # Remove low-confidence concepts
        # Merge similar concepts
        # Strengthen correlations
        pass
    
    def homeostatic_scaling(self):
        """Synaptic scaling to maintain stable activity"""
        # Ensure average activity remains in target range
        pass


class TemporalEncoder:
    """Encode temporal information (like positional encoding)"""
    
    def __init__(self, d_model=512):
        self.d_model = d_model
    
    def encode(self, time_ms):
        """
        Encode time as vector (sinusoidal encoding).
        
        Different frequencies capture different timescales.
        """
        position = time_ms
        encoding = torch.zeros(self.d_model)
        
        for i in range(0, self.d_model, 2):
            div_term = np.exp(i * -(np.log(10000.0) / self.d_model))
            encoding[i] = np.sin(position * div_term)
            if i + 1 < self.d_model:
                encoding[i + 1] = np.cos(position * div_term)
        
        return encoding
```

**Usage Example:**

```python
# Initialize timing system
timing_system = InternalTimingSystem(
    base_tick_ms=10,  # 100Hz base clock
    enable_circadian=True,
    enable_sleep_cycles=True
)

# During training loop
for step in range(training_steps):
    # Advance clock
    timing_system.tick()
    
    # Predict when reward will arrive
    expected_delay, confidence = timing_system.predict_event_time(
        event_type="reward_signal",
        context={"action": current_action}
    )
    
    # Schedule action at precise time
    timing_system.schedule_action(
        action=next_action,
        target_time_ms=timing_system.get_current_time() + 500,  # 500ms from now
        context={"task_difficulty": 0.7}
    )
    
    # Learn temporal patterns
    timing_system.learn_temporal_pattern(recent_event_sequence)
    
    # Check if sleep needed
    if timing_system.should_sleep():
        timing_system.enter_sleep_cycle()
        # Memory consolidation happens automatically

# Check current developmental phase
current_phase = timing_system.get_current_time(scale='phase')
print(f"Currently in Phase {current_phase}")
```

**Integration with Bio-Inspired Neurons:**

The timing system naturally integrates with bio-inspired neurons:

```python
# Neurons already have temporal dynamics
neuron = BioInspiredNeuron(
    input_dim=512,
    leak_rate=0.9,  # Leaky integration over time
    enable_stdp=True  # Spike-timing-dependent plasticity
)

# STDP learning window uses timing system
# Neurons that fire within 20ms strengthen connection
stdp_window_ms = 20
timing_system.temporal_encoder.encode(stdp_window_ms)

# Action timing precision improves with practice
for trial in range(100):
    target_time = 1000  # 1 second
    actual_time = execute_timed_action(neuron_output)
    error = target_time - actual_time
    
    # Learn timing adjustment
    timing_system.action_scheduler.record_timing(
        action="motor_command",
        target_time=target_time,
        actual_time=actual_time,
        success=(abs(error) < 50)  # Within 50ms = success
    )
```

**Benefits:**

1. **Multi-Scale Timing**: Milliseconds to months, all coordinated
2. **Temporal Prediction**: Learn "when" not just "what" and "how"
3. **Precise Action Timing**: Context-dependent timing adjustments
4. **Sleep Consolidation**: Automatic memory optimization during rest
5. **Developmental Tracking**: Phase transitions based on time + milestones
6. **Circadian Rhythms**: Activity patterns matching developmental stage
7. **Learned Patterns**: Discover temporal regularities in environment

**Developmental Progression:**

- **Phase 1**: Basic tick tracking, no prediction yet
- **Phase 2**: Learn simple temporal patterns (reward after action)
- **Phase 3**: Interval timing, schedule actions accurately
- **Phase 4**: Complex temporal prediction, first sleep cycles
- **Phase 5**: Full circadian rhythms, strategic sleep scheduling

#### Sensory-Grounded Thought System

Human thoughts are fundamentally grounded in sensory experiences—we think in images, sounds, feelings, not just abstract symbols. E-Brain's thoughts are similarly rooted in multi-sensory representations.

```python
class SensoryGroundedThoughtSystem:
    """
    Thoughts grounded in sensory modalities, like human cognition.
    
    Biological inspiration:
    - Mental imagery (visual thinking): "Picture a red apple"
    - Inner speech (auditory thinking): "What should I say?"
    - Tactile simulation (motor thinking): "How does it feel to grasp?"
    - Multimodal integration: Combine senses for rich thoughts
    - Sensory replay: Reactivate sensory cortices during thinking
    
    Key capabilities:
    - Thoughts have sensory components (visual, auditory, tactile, etc.)
    - Can "imagine" or "simulate" sensory experiences
    - Abstract concepts grounded in sensory experiences
    - Reasoning uses sensory simulation
    - Inner speech for language-based reasoning
    """
    
    def __init__(
        self,
        visual_encoder,
        audio_encoder,
        tactile_encoder,
        language_encoder,
        enable_mental_imagery=True
    ):
        # Sensory encoders (for grounding)
        self.visual_encoder = visual_encoder
        self.audio_encoder = audio_encoder
        self.tactile_encoder = tactile_encoder
        self.language_encoder = language_encoder
        
        # Sensory decoders (for simulation/imagination)
        self.visual_decoder = VisualImageryGenerator()
        self.audio_decoder = AuditorySimulator()
        self.tactile_decoder = TactilePredictor()
        
        # Multimodal integration
        self.multimodal_binder = MultimodalBinder()
        
        # Thought representation with sensory components
        self.current_thought = SensoryThought()
        
        # Mental imagery system
        self.mental_imagery_enabled = enable_mental_imagery
        self.imagery_buffer = []  # Visual mental images
        
        # Inner speech system
        self.inner_speech_buffer = []  # Auditory thoughts (words)
        
        # Sensory grounding database
        self.sensory_grounding_db = SensoryGroundingDatabase()
        
    def create_thought(self, task, context, modality_preferences=None):
        """
        Create a thought grounded in relevant sensory modalities.
        
        Args:
            task: Task description
            context: Current context
            modality_preferences: Which senses to emphasize ['visual', 'auditory', 'tactile']
        
        Returns:
            SensoryThought: Thought with sensory components
        """
        # Determine which sensory modalities are relevant
        if modality_preferences is None:
            modality_preferences = self._infer_relevant_modalities(task, context)
        
        thought = SensoryThought(task=task)
        
        # Add visual component if relevant
        if 'visual' in modality_preferences:
            visual_component = self._generate_visual_thought(task, context)
            thought.add_modality('visual', visual_component)
        
        # Add auditory component (inner speech)
        if 'auditory' in modality_preferences or 'language' in modality_preferences:
            auditory_component = self._generate_inner_speech(task, context)
            thought.add_modality('auditory', auditory_component)
        
        # Add tactile/motor component
        if 'tactile' in modality_preferences or 'motor' in modality_preferences:
            tactile_component = self._generate_tactile_prediction(task, context)
            thought.add_modality('tactile', tactile_component)
        
        # Add abstract/symbolic component
        symbolic_component = self._generate_symbolic_representation(task, context)
        thought.add_modality('symbolic', symbolic_component)
        
        # Bind modalities together
        thought.integrated_representation = self.multimodal_binder.bind(
            thought.modality_components
        )
        
        return thought
    
    def think_visually(self, concept):
        """
        Generate visual mental imagery for a concept.
        
        Example: "Think of a red apple"
        - Retrieves visual features from grounding database
        - Generates mental image using visual decoder
        """
        # Retrieve visual grounding
        visual_grounding = self.sensory_grounding_db.get_visual(concept)
        
        if visual_grounding is None:
            # No direct visual experience, try compositional
            visual_grounding = self._compose_visual_from_parts(concept)
        
        # Generate mental image
        mental_image = self.visual_decoder.generate(visual_grounding)
        self.imagery_buffer.append(mental_image)
        
        return mental_image
    
    def think_in_words(self, thought_content):
        """
        Inner speech: think in words/language.
        
        Example: "What should I say to the user?"
        - Converts thought to language
        - Simulates auditory representation (inner voice)
        """
        # Convert to language tokens
        language_tokens = self.language_encoder.tokenize(thought_content)
        
        # Generate inner speech (auditory simulation)
        inner_voice = self.audio_decoder.generate_speech(
            language_tokens,
            voice='inner'  # Own voice simulation
        )
        
        self.inner_speech_buffer.append(inner_voice)
        
        return inner_voice
    
    def imagine_action(self, action, context):
        """
        Simulate what an action would feel like (motor/tactile prediction).
        
        Example: "What happens if I move the block?"
        - Predicts tactile sensation
        - Predicts visual outcome
        - Uses for planning
        """
        # Tactile prediction
        tactile_prediction = self.tactile_decoder.predict(action, context)
        
        # Visual prediction (what will I see?)
        visual_prediction = self.visual_decoder.predict_next_frame(action, context)
        
        # Audio prediction (what will I hear?)
        audio_prediction = self.audio_decoder.predict_sound(action, context)
        
        simulated_experience = {
            'tactile': tactile_prediction,
            'visual': visual_prediction,
            'auditory': audio_prediction
        }
        
        return simulated_experience
    
    def ground_concept(self, concept, sensory_experiences):
        """
        Ground an abstract concept in sensory experiences.
        
        Example: "Dog" concept grounded in:
        - Visual: Four legs, fur, various breeds seen
        - Auditory: Barking sounds heard
        - Tactile: Soft fur when petted
        """
        grounding = ConceptGrounding(concept=concept)
        
        for experience in sensory_experiences:
            if experience.modality == 'visual':
                visual_features = self.visual_encoder(experience.data)
                grounding.add_visual(visual_features)
            
            elif experience.modality == 'auditory':
                audio_features = self.audio_encoder(experience.data)
                grounding.add_auditory(audio_features)
            
            elif experience.modality == 'tactile':
                tactile_features = self.tactile_encoder(experience.data)
                grounding.add_tactile(tactile_features)
        
        # Store grounding
        self.sensory_grounding_db.store(concept, grounding)
        
        return grounding
    
    def reason_with_imagery(self, problem):
        """
        Use mental imagery for spatial/visual reasoning.
        
        Example: "Can the couch fit through the door?"
        - Visualize couch dimensions
        - Visualize door dimensions
        - Mentally rotate/manipulate
        - Check if fits
        """
        # Generate mental images
        couch_image = self.think_visually("couch")
        door_image = self.think_visually("door")
        
        # Spatial reasoning using imagery
        can_fit = self._spatial_reasoning_with_images(
            couch_image,
            door_image,
            operation='fit_through'
        )
        
        return can_fit
    
    def multimodal_reasoning(self, task):
        """
        Combine multiple sensory modalities for reasoning.
        
        Example: "What made that sound?"
        - Auditory: Sound characteristics
        - Visual: Look for moving objects
        - Memory: Similar past experiences
        - Integration: "A car drove by"
        """
        # Collect sensory information
        auditory_info = self.get_current_audio()
        visual_info = self.get_current_visual()
        
        # Retrieve similar past experiences (multimodal)
        similar_experiences = self.sensory_grounding_db.retrieve_similar(
            auditory=auditory_info,
            visual=visual_info
        )
        
        # Integrate and infer
        hypothesis = self.multimodal_binder.integrate_and_infer(
            current={'auditory': auditory_info, 'visual': visual_info},
            past=similar_experiences
        )
        
        return hypothesis
    
    def _infer_relevant_modalities(self, task, context):
        """Infer which sensory modalities are relevant for task"""
        modalities = []
        
        task_lower = task.lower()
        
        # Visual tasks
        if any(word in task_lower for word in ['see', 'look', 'image', 'visual', 'color', 'shape']):
            modalities.append('visual')
        
        # Auditory/language tasks
        if any(word in task_lower for word in ['say', 'hear', 'sound', 'speak', 'listen', 'word']):
            modalities.append('auditory')
        
        # Tactile/motor tasks
        if any(word in task_lower for word in ['touch', 'feel', 'grasp', 'move', 'action', 'motor']):
            modalities.append('tactile')
        
        # Default: use all modalities if unclear
        if not modalities:
            modalities = ['visual', 'auditory', 'symbolic']
        
        return modalities
    
    def _generate_visual_thought(self, task, context):
        """Generate visual component of thought"""
        # Use visual decoder to imagine relevant imagery
        relevant_visual = self.visual_decoder.generate_task_relevant(task, context)
        return relevant_visual
    
    def _generate_inner_speech(self, task, context):
        """Generate inner speech (language-based thought)"""
        # Convert task to language representation
        inner_speech = self.language_encoder.encode(task)
        # Simulate auditory representation
        auditory_sim = self.audio_decoder.simulate_inner_voice(inner_speech)
        return auditory_sim
    
    def _generate_tactile_prediction(self, task, context):
        """Generate tactile/motor prediction"""
        # Predict what actions might feel like
        tactile_pred = self.tactile_decoder.predict_from_task(task, context)
        return tactile_pred
    
    def _generate_symbolic_representation(self, task, context):
        """Generate abstract symbolic representation"""
        # Traditional symbolic/vector representation
        return {'task_embedding': self.language_encoder(task)}
    
    def _compose_visual_from_parts(self, concept):
        """Compose visual representation from known parts"""
        # Example: "red apple" = red color + apple shape
        parts = self._decompose_concept(concept)
        visual_parts = [self.sensory_grounding_db.get_visual(p) for p in parts]
        composed = self.visual_decoder.compose(visual_parts)
        return composed
    
    def _spatial_reasoning_with_images(self, image1, image2, operation):
        """Spatial reasoning using mental imagery"""
        # Simplified: use visual reasoning network
        return self.visual_decoder.spatial_reasoning(image1, image2, operation)


class SensoryThought:
    """
    A thought with multiple sensory modality components.
    
    Like human thoughts: not just abstract symbols, but rich sensory experiences.
    """
    
    def __init__(self, task=None):
        self.task = task
        self.modality_components = {}
        self.integrated_representation = None
        self.creation_time = time.time()
    
    def add_modality(self, modality_name, component):
        """Add a sensory modality component to thought"""
        self.modality_components[modality_name] = component
    
    def get_modality(self, modality_name):
        """Retrieve specific sensory component"""
        return self.modality_components.get(modality_name)
    
    def has_modality(self, modality_name):
        """Check if thought has specific modality"""
        return modality_name in self.modality_components
    
    def get_dominant_modality(self):
        """Get the dominant sensory modality of this thought"""
        # Based on component strength/activation
        if not self.modality_components:
            return None
        
        # Simplified: return modality with highest activation
        dominant = max(
            self.modality_components.items(),
            key=lambda x: torch.norm(x[1]) if torch.is_tensor(x[1]) else 0
        )
        return dominant[0]
    
    def describe(self):
        """Describe thought in human-readable form"""
        desc = f"Thought about: {self.task}\n"
        desc += f"Modalities: {list(self.modality_components.keys())}\n"
        desc += f"Dominant: {self.get_dominant_modality()}\n"
        return desc


class SensoryGroundingDatabase:
    """
    Database linking abstract concepts to sensory experiences.
    
    Example:
    "Dog" → 
        Visual: [images of dogs]
        Auditory: [barking sounds]
        Tactile: [fur texture]
    """
    
    def __init__(self):
        self.visual_groundings = {}
        self.auditory_groundings = {}
        self.tactile_groundings = {}
        self.multimodal_index = {}
    
    def store(self, concept, grounding):
        """Store sensory grounding for concept"""
        if grounding.visual:
            self.visual_groundings[concept] = grounding.visual
        if grounding.auditory:
            self.auditory_groundings[concept] = grounding.auditory
        if grounding.tactile:
            self.tactile_groundings[concept] = grounding.tactile
        
        self.multimodal_index[concept] = grounding
    
    def get_visual(self, concept):
        """Get visual grounding for concept"""
        return self.visual_groundings.get(concept)
    
    def get_auditory(self, concept):
        """Get auditory grounding for concept"""
        return self.auditory_groundings.get(concept)
    
    def get_tactile(self, concept):
        """Get tactile grounding for concept"""
        return self.tactile_groundings.get(concept)
    
    def get_multimodal(self, concept):
        """Get all sensory groundings for concept"""
        return self.multimodal_index.get(concept)
    
    def retrieve_similar(self, **modality_queries):
        """
        Retrieve concepts with similar sensory properties.
        
        Example: retrieve_similar(auditory=barking_sound, visual=furry_thing)
        → Returns "dog", "wolf", etc.
        """
        candidates = []
        
        for concept, grounding in self.multimodal_index.items():
            similarity = 0.0
            num_modalities = 0
            
            if 'visual' in modality_queries and grounding.visual:
                sim = self._compute_similarity(
                    modality_queries['visual'],
                    grounding.visual
                )
                similarity += sim
                num_modalities += 1
            
            if 'auditory' in modality_queries and grounding.auditory:
                sim = self._compute_similarity(
                    modality_queries['auditory'],
                    grounding.auditory
                )
                similarity += sim
                num_modalities += 1
            
            if num_modalities > 0:
                avg_similarity = similarity / num_modalities
                candidates.append((concept, avg_similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:10]]
    
    def _compute_similarity(self, query, stored):
        """Compute similarity between sensory representations"""
        if torch.is_tensor(query) and torch.is_tensor(stored):
            return torch.cosine_similarity(query, stored, dim=-1).item()
        return 0.0


class ConceptGrounding:
    """Sensory grounding for a single concept"""
    
    def __init__(self, concept):
        self.concept = concept
        self.visual = []
        self.auditory = []
        self.tactile = []
        self.grounding_strength = 0.0
    
    def add_visual(self, visual_features):
        """Add visual experience"""
        self.visual.append(visual_features)
        self._update_strength()
    
    def add_auditory(self, audio_features):
        """Add auditory experience"""
        self.auditory.append(audio_features)
        self._update_strength()
    
    def add_tactile(self, tactile_features):
        """Add tactile experience"""
        self.tactile.append(tactile_features)
        self._update_strength()
    
    def _update_strength(self):
        """Update grounding strength based on # experiences"""
        total_experiences = len(self.visual) + len(self.auditory) + len(self.tactile)
        # More experiences = stronger grounding
        self.grounding_strength = min(1.0, total_experiences / 10.0)
    
    def get_prototypical_visual(self):
        """Get prototypical visual representation (average)"""
        if not self.visual:
            return None
        if torch.is_tensor(self.visual[0]):
            return torch.mean(torch.stack(self.visual), dim=0)
        return self.visual[0]
    
    def get_prototypical_auditory(self):
        """Get prototypical auditory representation"""
        if not self.auditory:
            return None
        if torch.is_tensor(self.auditory[0]):
            return torch.mean(torch.stack(self.auditory), dim=0)
        return self.auditory[0]


class VisualImageryGenerator:
    """
    Generate mental visual imagery (imagination).
    
    Like when you close your eyes and picture something.
    """
    
    def __init__(self, latent_dim=512):
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * 64 * 64),  # RGB 64x64 image
            nn.Tanh()
        )
    
    def generate(self, concept_embedding):
        """Generate mental image from concept"""
        if not torch.is_tensor(concept_embedding):
            concept_embedding = torch.randn(512)  # Placeholder
        
        mental_image = self.decoder(concept_embedding)
        mental_image = mental_image.view(-1, 3, 64, 64)
        return mental_image
    
    def generate_task_relevant(self, task, context):
        """Generate imagery relevant to task"""
        # Simplified: encode task and generate
        task_embedding = self._encode_task(task)
        return self.generate(task_embedding)
    
    def compose(self, visual_parts):
        """Compose visual image from parts"""
        # Average part embeddings
        if visual_parts:
            composed = torch.mean(torch.stack([v for v in visual_parts if v is not None]), dim=0)
            return self.generate(composed)
        return None
    
    def spatial_reasoning(self, image1, image2, operation):
        """Perform spatial reasoning on mental images"""
        # Simplified: use learned reasoning network
        # In practice: CNN for spatial operations
        return torch.rand(1).item() > 0.5  # Placeholder
    
    def predict_next_frame(self, action, context):
        """Predict what will be seen after action"""
        # Video prediction model
        return self.generate(torch.randn(512))  # Placeholder


class AuditorySimulator:
    """
    Simulate auditory experiences (inner speech, sound imagination).
    """
    
    def __init__(self):
        self.tts_model = None  # Text-to-speech for inner voice
        self.sound_generator = None
    
    def generate_speech(self, tokens, voice='inner'):
        """Generate inner speech (auditory thought)"""
        # Convert tokens to audio representation
        # Inner voice: own voice simulation
        audio_features = self._tokens_to_audio(tokens, voice)
        return audio_features
    
    def simulate_inner_voice(self, language_encoding):
        """Simulate hearing own thoughts"""
        # Language → auditory simulation
        return self._language_to_audio(language_encoding)
    
    def predict_sound(self, action, context):
        """Predict what sound action will make"""
        # Action → sound prediction
        return torch.randn(128)  # Audio features placeholder
    
    def _tokens_to_audio(self, tokens, voice):
        """Convert language tokens to audio features"""
        return torch.randn(128)  # Placeholder
    
    def _language_to_audio(self, encoding):
        """Convert language encoding to audio simulation"""
        return torch.randn(128)  # Placeholder


class TactilePredictor:
    """
    Predict tactile/motor sensations (how actions feel).
    """
    
    def __init__(self):
        self.prediction_network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def predict(self, action, context):
        """Predict tactile sensation from action"""
        # Encode action and context
        action_embedding = self._encode_action(action, context)
        tactile_prediction = self.prediction_network(action_embedding)
        return tactile_prediction
    
    def predict_from_task(self, task, context):
        """Predict relevant tactile info for task"""
        task_embedding = self._encode_task(task)
        return self.prediction_network(task_embedding)
    
    def _encode_action(self, action, context):
        """Encode action for prediction"""
        return torch.randn(512)  # Placeholder
    
    def _encode_task(self, task):
        """Encode task for prediction"""
        return torch.randn(512)  # Placeholder


class MultimodalBinder:
    """
    Bind multiple sensory modalities into unified representation.
    
    Inspired by brain's multimodal integration areas.
    """
    
    def __init__(self, output_dim=512):
        self.visual_proj = nn.Linear(2048, output_dim)
        self.audio_proj = nn.Linear(128, output_dim)
        self.tactile_proj = nn.Linear(128, output_dim)
        self.symbolic_proj = nn.Linear(512, output_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def bind(self, modality_components):
        """Bind multiple modalities into unified representation"""
        projections = []
        
        if 'visual' in modality_components:
            visual = modality_components['visual']
            if torch.is_tensor(visual):
                visual_flat = visual.flatten()
                visual_proj = self.visual_proj(visual_flat)
                projections.append(visual_proj)
        
        if 'auditory' in modality_components:
            audio = modality_components['auditory']
            if torch.is_tensor(audio):
                audio_proj = self.audio_proj(audio)
                projections.append(audio_proj)
        
        if 'tactile' in modality_components:
            tactile = modality_components['tactile']
            if torch.is_tensor(tactile):
                tactile_proj = self.tactile_proj(tactile)
                projections.append(tactile_proj)
        
        if 'symbolic' in modality_components:
            symbolic = modality_components['symbolic']['task_embedding']
            if torch.is_tensor(symbolic):
                symbolic_proj = self.symbolic_proj(symbolic)
                projections.append(symbolic_proj)
        
        # Pad if necessary
        while len(projections) < 4:
            projections.append(torch.zeros_like(projections[0]))
        
        # Concatenate and fuse
        concatenated = torch.cat(projections, dim=-1)
        unified = self.fusion(concatenated)
        
        return unified
    
    def integrate_and_infer(self, current, past):
        """Integrate current sensory info with past experiences"""
        # Bind current
        current_bound = self.bind(current)
        
        # Retrieve and bind similar past
        past_bound = [self.bind(p) for p in past[:5]]  # Top 5
        
        # Combine for inference
        if past_bound:
            combined = torch.mean(torch.stack([current_bound] + past_bound), dim=0)
        else:
            combined = current_bound
        
        return combined
```

**Usage Example:**

```python
# Initialize sensory-grounded thought system
sensory_system = SensoryGroundedThoughtSystem(
    visual_encoder=vision_encoder,
    audio_encoder=audio_encoder,
    tactile_encoder=tactile_encoder,
    language_encoder=language_encoder,
    enable_mental_imagery=True
)

# Example 1: Think visually about a concept
mental_image = sensory_system.think_visually("red apple")
# Generates internal visual representation (imagination)

# Example 2: Inner speech (think in words)
inner_voice = sensory_system.think_in_words("What should I tell the user?")
# Simulates hearing own thoughts

# Example 3: Imagine action before executing
simulated_experience = sensory_system.imagine_action(
    action="move_block_left",
    context=current_scene
)
# Returns: {tactile: "feels heavy", visual: "block moves left", auditory: "sliding sound"}

# Example 4: Reason with mental imagery
can_fit = sensory_system.reason_with_imagery("Can the couch fit through the door?")
# Uses visual mental manipulation

# Example 5: Ground abstract concept in sensory experiences
sensory_system.ground_concept(
    concept="dog",
    sensory_experiences=[
        SensoryExperience(modality='visual', data=dog_image),
        SensoryExperience(modality='auditory', data=bark_sound),
        SensoryExperience(modality='tactile', data=fur_texture)
    ]
)

# Example 6: Create multimodal thought
thought = sensory_system.create_thought(
    task="Describe what you see and hear",
    context=environment,
    modality_preferences=['visual', 'auditory']
)
print(thought.describe())
# Thought about: Describe what you see and hear
# Modalities: ['visual', 'auditory', 'symbolic']
# Dominant: visual

# Example 7: Multimodal reasoning
hypothesis = sensory_system.multimodal_reasoning("What made that sound?")
# Integrates: audio (sound characteristics) + visual (moving objects) → "A car drove by"
```

**Integration with Concurrent Thoughts:**

```python
# Thoughts now have sensory components
thought_1 = ThoughtStream(
    task="Solve math problem",
    sensory_components={
        'visual': mental_image_of_equation,
        'symbolic': abstract_representation,
        'auditory': inner_speech_counting
    }
)

# Different thoughts use different modalities
thought_2 = ThoughtStream(
    task="Listen to music",
    sensory_components={
        'auditory': music_representation,  # Dominant
        'visual': None,  # Not needed
        'symbolic': music_structure
    }
)

# Cross-pollination includes sensory insights
when thought_1 discovers_pattern():
    insight = {
        'content': "Sequence follows Fibonacci",
        'visual_pattern': visual_representation_of_sequence,
        'auditory': inner_speech("It's Fibonacci!")
    }
    shared_insight_memory.store(insight)
```

**Benefits:**

1. **Grounded Cognition**: Thoughts rooted in real sensory experiences
2. **Mental Imagery**: Can "imagine" or "visualize" during reasoning
3. **Inner Speech**: Think in words (language-based reasoning)
4. **Multimodal Integration**: Rich thoughts combining multiple senses
5. **Sensory Simulation**: Predict what actions will feel/look/sound like
6. **Concept Grounding**: Abstract concepts linked to concrete experiences
7. **Human-like Thinking**: Matches how humans actually think (in images, sounds, feelings)

**Developmental Progression:**

- **Phase 1**: Simple sensory associations (visual pattern → label)
- **Phase 2**: Basic sensory grounding (object → visual features)
- **Phase 3**: Inner speech emerges, visual mental imagery begins
- **Phase 4**: Rich multimodal thoughts, sensory simulation for planning
- **Phase 5**: Expert mental imagery, complex sensory reasoning

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
