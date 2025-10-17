"""
This module contains the implementation of the Double DQN agent.
Double DQN reduces overestimation bias by using the main network to select actions
and the target network to evaluate them.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple

from src.rl.config import DEVICE, AGENT
from src.rl.models.q_network import QNetwork

# Define a named tuple for storing transitions in replay memory for better batching
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class CircularTensorReplayBuffer:
    """
    Circular tensor-based replay buffer with two-phase storage.
    This implementation uses pre-allocated GPU memory slots.
    """
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        self.state_shape = state_shape
        
        # Pre-allocate tensors for the entire buffer
        self.states = torch.zeros((capacity,) + state_shape, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity,) + state_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        # Use a special marker value for terminal states in next_states
        # We'll use -999.0 in the first element to mark terminal states
        self.terminal_marker = -999.0
        
        self.current_experience_position = None  # Track current experience being built
        
    def start_experience(self, state, action):
        """
        Phase 1 of storing a memory: store the current state and action.
        Reward and next_state will be added later, after the transition is complete.

        Args:
            state: Current state tensor
            action: Action taken (integer)
        """
        # Store current experience position (to be used in phase 2)
        self.current_experience_position = self.position
        
        # Store current state and action
        self.states[self.position].copy_(state)
        self.actions[self.position] = action
        
    def complete_experience(self, reward, next_state):
        """
        Phase 2 of storing a memory: store the reward and next state.
        
        Args:
            reward: Reward received (float)
            next_state: Next state tensor or None if terminal
        """
        if self.current_experience_position is None:
            raise RuntimeError("Must call start_experience() before complete_experience()")
            
        pos = self.current_experience_position
        
        # Store reward
        self.rewards[pos] = reward
        
        if next_state is not None:
            # Store next state
            self.next_states[pos].copy_(next_state)
        else:
            # Terminal state: mark with special value in first element
            self.next_states[pos].fill_(0.0)
            self.next_states[pos, 0, 0, 0] = self.terminal_marker
            
        # Complete the experience and advance buffer
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.current_experience_position = None
        
    def sample(self, batch_size, generator=None):
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            generator: PyTorch random generator for sampling
            
        Returns:
            Tuple of (states, actions, rewards, next_states, non_final_mask)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device, generator=generator)
        
        # Compute non_final_mask by checking for terminal marker
        non_final_mask = self.next_states[indices, 0, 0, 0] != self.terminal_marker
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            non_final_mask
        )
    
    def __len__(self):
        """Return current number of experiences in buffer."""
        return self.size


class DQNAgent:
    def __init__(self, width, height, gamma=None, epsilon=None, epsilon_min=None, 
                 epsilon_decay=None, batch_size=None, memory_size=None, channels=None,
                 agent_seed=None, lr=None, polyak_tau=None, grad_clip=None):
        """
        Initialize the DQN agent with parameters from config.
        
        Args:
            width: Width of the grid environment
            height: Height of the grid environment
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate of epsilon (higher means slower decay)
            batch_size: Size of batch for experience replay
            memory_size: Size of replay buffer
            channels: Number of state channels
            agent_seed: Random seed for agent's internal randomness
            lr: Learning rate
            polyak_tau: Averaging parameter for soft target updates
            grad_clip: Maximum gradient norm for gradient clipping
        """
        # Use provided values or fall back to config values
        self.gamma = gamma if gamma is not None else AGENT["gamma"]
        self.epsilon = epsilon if epsilon is not None else AGENT["epsilon"]
        self.epsilon_min = epsilon_min if epsilon_min is not None else AGENT["epsilon_min"]
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else AGENT["epsilon_decay"]
        self.polyak_tau = polyak_tau if polyak_tau is not None else AGENT["polyak_tau"]
        self.steps_done = 0
        self.batch_size = batch_size if batch_size is not None else AGENT["batch_size"]
        self.agent_seed = agent_seed if agent_seed is not None else AGENT["agent_seed"]
        self.grad_clip = grad_clip if grad_clip is not None else AGENT["grad_clip"]
        channels = channels if channels is not None else AGENT["channels"]
        memory_size = memory_size if memory_size is not None else AGENT["memory_size"]
        lr = lr if lr is not None else AGENT["lr"]
        
        self.grid_size = (width, height)
        
        # Initialize Q-network and target Q-network
        self.q_network = QNetwork(
            width, 
            height, 
            channels
        ).to(DEVICE)
        
        self.target_network = QNetwork(
            width, 
            height, 
            channels
        ).to(DEVICE)
        
        # Copy initial weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize circular tensor buffer for experience replay
        state_shape = (width, height, channels)
        self.replay_buffer = CircularTensorReplayBuffer(memory_size, state_shape, DEVICE)

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-5)
        self.loss_fn = torch.nn.MSELoss()
        
        # Set random seed for reproducibility
        self.rng = np.random.default_rng(seed=self.agent_seed)  # Dedicated NumPy RNG for CPU operations
        self.torch_generator = torch.Generator(device=DEVICE)  # Dedicated PyTorch RNG for GPU operations
        self.torch_generator.manual_seed(self.agent_seed)

    def act(self, state, exploration=True, return_q_values=False, action_set_mask=None):
        """
        Choose an action using the Q-network
        
        Args:
            state: Current state tensor
            exploration: Whether to use exploration
            return_q_values: Whether to return Q-values for visualization
            action_set_mask: Optional mask to filter valid actions given the action set
            
        Returns:
            If return_q_values is False: A tuple (x, y) representing the action coordinates
            If return_q_values is True: A tuple ((x, y), q_values_grid) where q_values_grid is a 2D numpy array
        """
        
        # Calculate epsilon (exploration parameter) with exponential decay
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if exploration:
            self.steps_done += 1
        
        # Get Q values (already set to -inf for invalid actions)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            q_values = q_values.squeeze()

        # Apply action set mask if provided
        if action_set_mask is not None:
            action_set_mask = torch.tensor(action_set_mask, device=DEVICE, dtype=torch.bool).flatten()
            q_values[~action_set_mask] = -torch.inf
        if self.rng.random() < eps_threshold and exploration:
            # With probability eps_threshold, sample from valid actions
            valid_mask = torch.isfinite(q_values)
            uniform_probs = valid_mask.float() / valid_mask.sum()
            action_index = torch.multinomial(uniform_probs, 1, generator=self.torch_generator).item()
        else:
            # Greedy: take the argmax of the Q values
            action_index = torch.argmax(q_values).item()
        
        x, y = divmod(action_index, self.grid_size[1])
        
        if return_q_values:
            # Convert Q-values to 2D grid format for visualization
            q_values_grid = q_values.cpu().numpy().reshape(self.grid_size[0], self.grid_size[1])
            # Replace -inf values with NaN for better visualization
            q_values_grid = np.where(np.isfinite(q_values_grid), q_values_grid, np.nan)
            return (x, y), q_values_grid
        else:
            return x, y

    def replay(self):
        """
        Learn from experience replay with batched processing using Double DQN.
        Double DQN uses the main network to select the best action and the target 
        network to evaluate that action, reducing overestimation bias.
        
        Returns:
            Tuple of (loss, max_predicted_q, max_next_state_value, mean_abs_td_error)
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask = \
            self.replay_buffer.sample(self.batch_size, self.torch_generator)
        action_batch = action_batch.unsqueeze(1)
        
        # Compute the Q values for state s_t and select the ones corresponding to the taken actions
        all_q_values = self.q_network(state_batch)
        state_action_values = all_q_values.gather(1, action_batch).squeeze(1)
        
        # Compute Q-value statistics from all Q-values
        with torch.no_grad():
            # Flatten all Q-values and filter out -inf (masked actions)
            flat_q_values = all_q_values.flatten()
            finite_q_values = flat_q_values[torch.isfinite(flat_q_values)]
            
            if len(finite_q_values) > 0:
                q_stats = {
                    'mean': finite_q_values.mean().item(),
                    'std': finite_q_values.std().item(),
                    'min': finite_q_values.min().item(),
                    'max': finite_q_values.max().item()
                }
            else:
                q_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        # Compute V(s_{t+1}) for all next states using Double DQN
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        
        with torch.no_grad():
            # Double DQN: Use main network to select actions, target network to evaluate them
            main_next_q_values_all = self.q_network(next_state_batch)
            best_actions = main_next_q_values_all.max(dim=1)[1]
            
            # Evaluate selected actions using target network
            target_next_q_values_all = self.target_network(next_state_batch)
            target_next_q_values = target_next_q_values_all.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Only assign Q-values for non-terminal states (terminal states remain 0)
            next_state_values[non_final_mask] = target_next_q_values[non_final_mask]
            
            # Compute expected Q values
            expected_state_action_values = reward_batch + non_final_mask.float() * self.gamma * next_state_values
        
        # Track the key values for monitoring training progress
        max_predicted_q = torch.max(state_action_values).item()  
        max_next_state_value = torch.max(next_state_values).item() if next_state_values.numel() > 0 else 0.0
        td_error = expected_state_action_values - state_action_values
        mean_abs_td_error = torch.mean(torch.abs(td_error)).item()
        
        # Compute loss and update weights
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.grad_clip)
        
        self.optimizer.step()
        
        # Soft update of target network
        self.soft_update_target_network()

        return loss.item(), max_predicted_q, max_next_state_value, mean_abs_td_error, q_stats
    
    def soft_update_target_network(self):
        """
        Soft update target network using Polyak averaging.
        """
        tau = self.polyak_tau
        one_minus_tau = 1.0 - tau
        
        with torch.no_grad():
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(one_minus_tau).add_(local_param.data, alpha=tau)

    def save_model(self, filename="dqn_model.pth"):
        """Save the agent's model to a file."""
        memory_dict = {
            'states': self.replay_buffer.states[:self.replay_buffer.size].cpu(),
            'next_states': self.replay_buffer.next_states[:self.replay_buffer.size].cpu(),
            'actions': self.replay_buffer.actions[:self.replay_buffer.size].cpu(),
            'rewards': self.replay_buffer.rewards[:self.replay_buffer.size].cpu(),
            'position': self.replay_buffer.position,
            'size': self.replay_buffer.size,
            'capacity': self.replay_buffer.capacity,
            'terminal_marker': self.replay_buffer.terminal_marker
        }
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory': memory_dict
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="dqn_model_final.pth"):
        """Load the agent's model from a file."""
        checkpoint = torch.load(filename, map_location=DEVICE)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint.get('steps_done', 0)
        
        # Restore memory if available
        if 'memory' in checkpoint:
            memory_dict = checkpoint['memory']
            size = memory_dict['size']
            self.replay_buffer.states[:size].copy_(memory_dict['states'].to(DEVICE))
            self.replay_buffer.next_states[:size].copy_(memory_dict['next_states'].to(DEVICE))
            self.replay_buffer.actions[:size].copy_(memory_dict['actions'].to(DEVICE))
            self.replay_buffer.rewards[:size].copy_(memory_dict['rewards'].to(DEVICE))
            self.replay_buffer.position = memory_dict['position']
            self.replay_buffer.size = size
            if 'terminal_marker' in memory_dict:
                self.replay_buffer.terminal_marker = memory_dict['terminal_marker']
            print(f"Restored {size} experiences from circular buffer")
            
        print(f"Model loaded from {filename}")
    
    def get_current_epsilon(self):
        """
        Return the current epsilon value after decay.
        """
        # Calculate epsilon with exponential decay, same formula as in act()
        return self.epsilon_min + (self.epsilon - self.epsilon_min) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)

    def reset(self, seed=None):
        """
        Reset the agent's random generator.
        """
        if seed is not None:
            self.agent_seed = int(seed)
        self.rng = np.random.default_rng(seed=self.agent_seed)
        self.torch_generator.manual_seed(self.agent_seed)