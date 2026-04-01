from DQN_agent import DQAgent
import gymnasium as gym
from transforms import Transforms
import numpy as np
import ale_py

# Specify environment location
env_name = "ALE/Breakout-v5"

# Initializes an openai gym environment

def init_gym_env(env_path):
    # 1. Ajout du render_mode pour l'affichage visuel
    env = gym.make(env_path, render_mode="human")

    # 2. Déballage du tuple (nouvelle version de Gym)
    state_initial, _ = env.reset()
    
    # 3. L'espace d'état sera composé de 4 images de taille 84x84
    # On force manuellement cette dimension pour que le réseau s'initialise correctement
    state_space = (4, 84, 84) 
    action_space = env.action_space.n

    return env, state_space, action_space

# Initialize Gym Environment
env, state_space, action_space = init_gym_env(env_name)
    
# Create an agent
agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=action_space, model_name='breakout_model', gamma=.99,
                eps_strt=1.0, eps_end=.001, eps_dec=1e-3, batch_size=32, lr=.001)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75000) ##75000

#agent.play_games(num_eps=7, render=True)