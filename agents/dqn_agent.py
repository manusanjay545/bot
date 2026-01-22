"""
Deep Q-Network (DQN) Agent with Dueling architecture and Double DQN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DQN_CONFIG, MODELS_DIR
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DuelingDQN(nn.Module):
    """Dueling DQN Network Architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = DQN_CONFIG['hidden_layers']
        
        # Feature extraction layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1] // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1] // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage (Dueling architecture)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """DQN Agent with Double DQN and Prioritized Experience Replay"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
        batch_size: int = None,
        memory_size: int = None,
        target_update: int = None,
        use_prioritized_replay: bool = True,
        device: str = None
    ):
        # Config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate or DQN_CONFIG['learning_rate']
        self.gamma = gamma or DQN_CONFIG['gamma']
        self.epsilon = epsilon_start or DQN_CONFIG['epsilon_start']
        self.epsilon_end = epsilon_end or DQN_CONFIG['epsilon_end']
        self.epsilon_decay = epsilon_decay or DQN_CONFIG['epsilon_decay']
        self.batch_size = batch_size or DQN_CONFIG['batch_size']
        self.target_update = target_update or DQN_CONFIG['target_update']
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay buffer
        memory_size = memory_size or DQN_CONFIG['memory_size']
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)
        self.use_prioritized = use_prioritized_replay
        
        # Training stats
        self.train_step = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step_update(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        if self.use_prioritized:
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(len(states)).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Compute loss
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        loss = (weights * (current_q - target_q).pow(2).squeeze()).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized:
            self.memory.update_priorities(indices, td_errors)
        
        self.train_step += 1
        
        # Update target network
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """Called at end of each episode"""
        self.episode_count += 1
        self.decay_epsilon()
    
    def save(self, filepath: str = None):
        """Save model weights"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f'dqn_agent_{self.episode_count}.pt')
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'train_step': self.train_step
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.train_step = checkpoint['train_step']
        print(f"Model loaded from {filepath}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy().flatten()
