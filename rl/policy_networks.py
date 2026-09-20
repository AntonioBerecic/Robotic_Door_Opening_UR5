import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        if isinstance(state_space, int): # pass in state_dim rather than state_space
            self._state_dim = state_space
        else:
            self._state_space = state_space
            self._state_shape = state_space.shape
            if len(self._state_shape) == 1:
                self._state_dim = self._state_shape[0]
            else:  # high-dim state
                pass  
        
        if isinstance(action_space, int): # pass in action_dim rather than action_space
            self._action_dim = action_space
        else:
            self._action_space = action_space
            self._action_shape = action_space.shape
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()


class DPG_PolicyNetwork(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3, machine_type='gpu'):
        super().__init__(state_space, action_space, action_range)
        self.machine_type = machine_type
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim) 
        self.ln4 = nn.LayerNorm(hidden_dim)

        torch.nn.init.orthogonal_(self.linear1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.linear2.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.linear3.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.linear4.weight, gain=np.sqrt(2))

        self.output_linear = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action
        # weights initialization
        self.output_linear.weight.data.uniform_(-init_w, init_w)
        self.output_linear.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, hidden_activation=F.relu, output_activation=F.tanh):
        x = hidden_activation(self.ln1(self.linear1(state)))
        x = hidden_activation(self.ln2(self.linear2(x)))
        x = hidden_activation(self.ln3(self.linear3(x)))
        x = hidden_activation(self.ln4(self.linear4(x)))
        output  = output_activation(self.output_linear(x))
        return output

    def evaluate(self, state, noise_scale=0.5):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        action = self.forward(state)
        ''' add noise '''
        normal = Normal(0, 1)
        eval_noise_clip = 2*noise_scale
        device = next(self.parameters()).device
        noise = normal.sample(action.shape) * noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = self.action_range*action + noise.to(device)
        return torch.clamp(action, -self.action_range, self.action_range)


    def get_action(self, state, noise_scale=0.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        device = next(self.parameters()).device
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0] 
        ''' add noise '''
        normal = Normal(0, 1)
        noise = noise_scale * normal.sample(action.shape)
        action=self.action_range*action + noise.numpy()
        return np.clip(action, -self.action_range, self.action_range)
