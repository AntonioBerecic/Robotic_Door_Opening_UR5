'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
'''
import math
import random
import gym
import numpy as np
import torch
from torch.distributions import Normal
import queue

from rl.optimizers import SharedAdam, ShareParameters
from rl.buffers import ReplayBuffer
from rl.value_networks import QNetwork
from rl.policy_networks import DPG_PolicyNetwork
from utils.load_params import load_params
from utils.common_func import rand_params
import os
import copy

from mujoco_py import MujocoException

###############################  TD3  ####################################

class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        action_range, policy_target_update_interval=1, machine_type='gpu'):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim
        self.machine_type = machine_type
        self.device = torch.device(
            'cuda' if machine_type == 'gpu' and torch.cuda.is_available() else 'cpu')

        self.q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.q_net2 = QNetwork(state_space, action_space, hidden_dim)
        # self.target_q_net1 = copy.deepcopy(self.q_net1)
        # self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_q_net1 = QNetwork(state_space, action_space, hidden_dim)
        self.target_q_net2 = QNetwork(state_space, action_space, hidden_dim)
        self.policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range, machine_type=machine_type)
        # self.target_policsy_net = copy.deepcopy(self.policy_net)
        self.target_policy_net = DPG_PolicyNetwork(state_space, action_space, hidden_dim, action_range, machine_type=machine_type)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        # Move parameters before optimizers are constructed. Moving a module
        # afterwards can leave an optimizer referring to the old parameters.
        self.q_net1.to(self.device)
        self.q_net2.to(self.device)
        self.target_q_net1.to(self.device)
        self.target_q_net2.to(self.device)
        self.policy_net.to(self.device)
        self.target_policy_net.to(self.device)
    
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = SharedAdam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = SharedAdam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = SharedAdam(self.policy_net.parameters(), lr=policy_lr)

    def to_cuda(self):
        if self.device.type != 'cuda':
            raise RuntimeError(
                "TD3_Trainer must be constructed with machine_type='gpu'; "
                "moving it after optimizer creation is unsafe")
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        for _ in range(3): # sample several times to prevent unkown error of failure in sampling
            try:
                state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
                break
            except Exception as e:
                print(e)
            
        state      = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action     = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward     = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done       = torch.as_tensor(np.float32(done), dtype=torch.float32,
                                     device=self.device).unsqueeze(1)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action = self.policy_net.evaluate(state, noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action = self.target_policy_net.evaluate(next_state, noise_scale=eval_noise_scale) # clipped normal noise

        if reward_scale:
            reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),self.target_q_net2(next_state, new_next_action))

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        if torch.isnan(q_value_loss1): # error capture
            print('Error: q loss 1 value is nan')
            # print(state, action, reward, next_state, done)
            # breakpoint()
        else:
            self.q_optimizer1.zero_grad()
            q_value_loss1.backward()
            self.q_optimizer1.step()
        if torch.isnan(q_value_loss2): # error captur
            print('Error: q loss 2 value is nan')
            # breakpoint()
        else:
            self.q_optimizer2.zero_grad()
            q_value_loss2.backward()
            self.q_optimizer2.step()

        if self.update_cnt%self.policy_target_update_interval==0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()
            if torch.isnan(policy_loss): # error capture
                print('Error: policy loss value is nan')
                breakpoint()
            else:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
            
            # Soft update the target nets
            self.target_q_net1=self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2=self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt+=1

        return predicted_q_value1.mean()

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1', map_location=self.device))
        self.q_net2.load_state_dict(torch.load(path+'_q2', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=self.device))
        # A loaded online model must not continue with randomly initialized
        # targets; this is especially important when fine-tuning.
        self.target_ini(self.q_net1, self.target_q_net1)
        self.target_ini(self.q_net2, self.target_q_net2)
        self.target_ini(self.policy_net, self.target_policy_net)
        # self.q_net1.eval()
        # self.q_net2.eval()
        # self.policy_net.eval()

    def share_memory(self):
        self.q_net1.share_memory()
        self.q_net2.share_memory()
        self.target_q_net1.share_memory()
        self.target_q_net2.share_memory()
        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        ShareParameters(self.q_optimizer1)
        ShareParameters(self.q_optimizer2)
        ShareParameters(self.policy_optimizer)


def collector_worker(id, policy, policy_lock, episode_counter, envs, env_name,
        rewards_queue, replay_buffer, max_episodes, max_steps, explore_steps,
        num_workers, noise_decay, explore_noise_scale, action_range, render,
        randomized_params, env_kwargs=None, seed=1):
    """Collect experience on CPU; network optimization belongs to the learner."""
    worker_seed = seed + id + 1
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    env_kwargs = dict(env_kwargs or {})
    env = envs[env_name](**env_kwargs)
    frame_idx = 0
    # Distribute the initial random exploration budget across collectors.
    worker_explore_steps = int(math.ceil(float(explore_steps) / num_workers))

    for eps in range(max_episodes):
        episode_reward = 0.0
        episode_success = False
        episode_max_door_angle = 0.0
        episode_min_knob_distance = float('inf')
        episode_min_orientation_error = float('inf')
        with episode_counter.get_lock():
            global_episode = episode_counter.value
            episode_counter.value += 1
        current_noise = explore_noise_scale * (noise_decay ** global_episode)

        if randomized_params:
            state = env.reset(**(rand_params(env, params=randomized_params)[0]))
        else:
            state = env.reset()

        for step in range(max_steps):
            assert np.all(np.isfinite(state)), "Non-finite state detected"
            if frame_idx < worker_explore_steps:
                action = np.random.uniform(
                    -action_range, action_range, size=policy._action_dim)
            else:
                # Prevent the learner from publishing parameters halfway through
                # a collector forward pass.
                with policy_lock:
                    action = policy.get_action(state, noise_scale=current_noise)
            action = np.clip(action, -action_range, action_range)

            try:
                next_state, reward, done, info = env.step(action)
                episode_success = episode_success or bool(info.get('success', False))
                episode_max_door_angle = max(
                    episode_max_door_angle,
                    float(info.get('door_open_angle', 0.0)))
                episode_min_knob_distance = min(
                    episode_min_knob_distance,
                    float(info.get('distance_to_knob', float('inf'))))
                episode_min_orientation_error = min(
                    episode_min_orientation_error,
                    float(info.get('orientation_error', float('inf'))))
                if render:
                    env.render()
            except MujocoException:
                print('Worker:', id, '| MujocoException; recreating environment')
                env = envs[env_name](**env_kwargs)
                break

            values = [np.sum(state), np.sum(action), reward, np.sum(next_state)]
            if np.all(np.isfinite(values)):
                replay_buffer.push(state, action, reward, next_state, done)
            else:
                print('Worker:', id, '| Non-finite transition skipped')

            state = next_state
            episode_reward += reward
            frame_idx += 1
            if done:
                break

        print('Worker:', id, '|Episode:', eps, '| Episode Reward:',
              episode_reward, '| Step:', step, '| Success:', episode_success,
              '| Min knob distance:', episode_min_knob_distance,
              '| Min orientation error:', episode_min_orientation_error,
              '| Max door angle:', episode_max_door_angle)
        rewards_queue.put((episode_reward, episode_success, step + 1))
