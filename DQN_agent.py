import torch
import torch.nn as nn
import numpy as np
import random
import math
from collections import deque
from replay_mem import ReplayBuffer
from DQN_model import DQN
from transforms import Transforms
from PIL import Image


class DQAgent(object):
    def __init__(self, replace_target_cnt, env, state_space, action_space, 
                model_name='breakout_model', gamma=0.99, eps_strt=0.1, 
                eps_end=0.001, eps_dec=5e-6, batch_size=32, lr=0.001):

        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.GAMMA = gamma
        self.LR = lr
        self.eps = eps_strt
        self.eps_dec = eps_dec
        self.eps_end = eps_end

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.memory = ReplayBuffer()

        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        self.policy_net = DQN(self.state_space, self.action_space, filename=model_name).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename=model_name+'target').to(self.device)
        self.target_net.eval()

        try:
            self.policy_net.load_model()
            print('loaded pretrained model')
        except:
            pass
        
        self.replace_target_net()

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.SmoothL1Loss()

    # -------------------- BATCH --------------------
    def sample_batch(self):
        batch = self.memory.sample_batch(self.batch_size)

        state = torch.tensor(np.array(batch.state) / 255.0).float().to(self.device)
        state_ = torch.tensor(np.array(batch.state_) / 255.0).float().to(self.device)
        action = torch.tensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward = torch.tensor(np.array(batch.reward)).float().unsqueeze(1).to(self.device)
        done = torch.tensor(np.array(batch.done)).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # -------------------- ACTION --------------------
    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action = self.policy_net(obs).argmax().item()
        return action

    def choose_action(self, obs, n_passes=10, threshold=0.4):
        if random.random() < self.eps:
            return random.randrange(self.action_space)

        obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(self.device)

        self.policy_net.train()
        q_values_list = []

        with torch.no_grad():
            for _ in range(n_passes):
                q_values_list.append(self.policy_net(obs_tensor))

        q_values = torch.cat(q_values_list, dim=0)

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)

        best_action = q_mean.argmax().item()
        uncertainty = q_std[best_action].item()

        if uncertainty < threshold:
            return best_action
        else:
            return random.randrange(self.action_space)

    # -------------------- MEMORY --------------------
    def store_transition(self, *args):
        self.memory.add_transition(*args)

    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('Target network replaced')

    def dec_eps(self):
        self.eps = max(self.eps - self.eps_dec, self.eps_end)

    # -------------------- LEARNING --------------------
    def learn(self, num_iters=1):
        if self.memory.pointer < self.batch_size:
            return 

        for _ in range(num_iters):

            state, action, reward, state_, done = self.sample_batch()

            # Q(s,a)
            q_eval = self.policy_net(state).gather(1, action)

            # 🔥 DOUBLE DQN
            self.policy_net.eval()
            next_actions = self.policy_net(state_).argmax(1).unsqueeze(1)
            self.policy_net.train()

            q_next = self.target_net(state_).gather(1, next_actions).detach()

            q_target = (1 - done) * (reward + self.GAMMA * q_next) + (done * reward)

            loss = self.loss(q_eval, q_target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.learn_counter += 1
            self.replace_target_net()

        self.policy_net.save_model()
        self.dec_eps()

    # -------------------- TRAIN --------------------
    def train(self, num_eps=100, render=True):
        scores = []
        stack_size = 4

        for i in range(num_eps):
            obs, _ = self.env.reset()
            obs, _, _, _, _ = self.env.step(1)  # FIRE

            processed = np.squeeze(Transforms.to_gray(obs))
            frames = deque([processed]*stack_size, maxlen=stack_size)

            state = np.stack(frames, axis=0)

            done = False
            score = 0
            steps = 0

            while not done:
                action = self.choose_action(state)
                obs_, reward, terminated, truncated, _ = self.env.step(action)

                reward = np.clip(reward, -1, 1)
                done = terminated or truncated

                if render:
                    self.env.render()

                processed_ = np.squeeze(Transforms.to_gray(obs_))
                frames.append(processed_)

                state_ = np.stack(frames, axis=0)

                self.store_transition(state, action, reward, state_, int(done), obs)

                state = state_
                obs = obs_

                score += reward
                steps += 1

            scores.append(score)

            print(f'Episode {i}/{num_eps} | Score: {score} | Avg: {np.mean(scores[-100:])} | Eps: {self.eps}')

            self.learn(math.ceil(steps / self.batch_size))

        self.env.close()

    # -------------------- PLAY --------------------
    def play_games(self, num_eps, render=True):
        self.policy_net.eval()

        for i in range(num_eps):
            obs, _ = self.env.reset()
            obs, _, _, _, _ = self.env.step(1)

            frames = deque([np.squeeze(Transforms.to_gray(obs))]*4, maxlen=4)
            state = np.stack(frames, axis=0)

            done = False
            score = 0

            while not done:
                action = self.greedy_action(state)
                obs_, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                if render:
                    self.env.render()

                processed_ = np.squeeze(Transforms.to_gray(obs_))
                frames.append(processed_)

                state = np.stack(frames, axis=0)
                obs = obs_

                score += reward

            print(f'Episode {i}: Score = {score}')

        self.env.close()