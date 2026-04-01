import numpy as np
from collections import namedtuple

# Création d'un namedtuple pour extraire facilement les données des batchs
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_', 'done'))

class ReplayBuffer(object):
    def __init__(self, max_size=50000, prioritized=True):
        self.max_size = max_size
        self.prioritized = prioritized
        self.memory = []
        self.pointer = 0
        
        # Hyperparamètres PER
        self.alpha = 0.6  # Niveau de priorisation (0 = aléatoire, 1 = total)
        self.beta = 0.4   # Importance-sampling
        self.beta_increment = 0.001
        self.eps = 1e-6   # Petite valeur pour éviter une priorité de zéro
        
        # Tableau pré-alloué pour les priorités (pour la rapidité)
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def add_transition(self, state, action, reward, state_, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        transition = Transition(state, action, reward, state_, done)
        
        if len(self.memory) < self.max_size:
            self.memory.append(transition)
        else:
            self.memory[self.pointer] = transition
        
        self.priorities[self.pointer] = max_prio
        self.pointer = (self.pointer + 1) % self.max_size

    def sample_batch(self, batch_size):
        if len(self.memory) == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pointer]
        
        if self.prioritized:
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probs)
        else:
            indices = np.random.choice(len(self.memory), batch_size, replace=False)
            probs = np.ones(len(self.memory)) / len(self.memory)
        
        samples = [self.memory[idx] for idx in indices]
        
        # Calcul des poids Importance-Sampling (IS)
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalisation
        
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        # Décompression du batch
        batch = Transition(*zip(*samples))
        
        return batch, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + self.eps