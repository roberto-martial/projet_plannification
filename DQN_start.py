from DQN_agent import DQAgent
import gym
from transforms import Transforms
import numpy as np

# Specify environment location
env_name = 'BreakoutNoFrameskip-v4'

# Initializes an openai gym environment
# Initializes an openai gym environment
# Initializes an openai gym environment
def init_gym_env(env_path):

    env = gym.make(env_path)

    # 1. On récupère l'état initial (et on déballe le tuple pour les nouvelles versions de Gym)
    state_initial, _ = env.reset()
    
    # 2. On garde la forme originale (210, 160, 3) pour créer notre tableau vide NumPy
    state_raw = np.zeros(state_initial.shape, dtype=np.uint8)
    
    # 3. On passe l'image au format (H, W, C) à Torchvision
    processed_state = Transforms.to_gray(state_raw)
    
    state_space = processed_state.shape
    action_space = env.action_space.n

    return env, state_space, action_space

# Initialize Gym Environment
env, state_space, action_space = init_gym_env(env_name)
    
# Create an agent
agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=action_space, model_name='breakout_model', gamma=.99,
                eps_strt=.1, eps_end=.001, eps_dec=5e-6, batch_size=32, lr=.001)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75) ##75000