import gin

import numpy as np
import torch
from torchinfo import summary as torch_summary
import random

import torch.nn.functional as F
from torch.optim import Adam
from torch import nn


from utils import device, get_free_mem, delete_these
from basics.abstract_algorithms import RecurrentOffPolicyRLAlgorithm
from basics.utils import create_target, mean_of_unmasked_elements, polyak_update
from buffer import RecurrentReplayBuffer




















class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

hidden_size = 128

@gin.configurable(module=__name__)
class Summarizer(nn.Module):
    
    def __init__(self, rgbd_input = (16,16,4)):
        super().__init__()
    
        example = torch.zeros(rgbd_input).unsqueeze(0).permute(0,3,1,2)
      
        self.cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4, 
                out_channels = 8,
                kernel_size = (3,3),
                padding = (1,1)
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1))
        )
      
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1] + 4 # Plus speed, energy, previous action
    
        self.lstm = nn.LSTM(
            input_size = quantity,
            hidden_size = hidden_size,
            batch_first = True)
      
        self.cnn.apply(init_weights)
        self.lstm.apply(init_weights)
        self.float()

    def forward(self, state, speed, energy, action = None, hidden = None, return_hidden=False):
        if(len(state.shape) == 3):     state = state.unsqueeze(0)
        if(len(state.shape) == 4):
            state = state.float().to(device).permute(0,3,1,2)
            state = self.cnn(state).flatten(1).unsqueeze(1)
        else:
            state = state.float().to(device).permute(1,0,4,2,3)
            state = torch.stack([self.cnn(step).flatten(1).unsqueeze(1) for step in state]).squeeze(2)
            state = state.permute(1, 0, 2)            
        if(type(speed) != torch.Tensor):
            speed = torch.tensor(speed)
        speed = F.relu(speed).to(device)
        if(len(speed.shape) != 3): speed = speed.view(state.shape[0],1,1)
        if(type(energy) != torch.Tensor):
            energy = torch.tensor(energy)
        energy = F.relu(energy).to(device) # / 100
        if(len(energy.shape) != 3): energy = energy.view(state.shape[0],1,1)
        while(len(action.shape) != 3):
            action = action.unsqueeze(0)
        state = torch.cat([state, speed, energy, action.to(device)], -1)
        self.lstm.flatten_parameters()
        if(hidden == None): state, hidden = self.lstm(state)
        else:               state, hidden = self.lstm(state.float(), (hidden[0].float(), hidden[1].float()))
        if(state.shape[1] == 1):
            state = state[:,-1,:]
            state = state.flatten(1)
        summary = F.relu(state)
        
        if return_hidden: return summary, hidden
        else:             return summary



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
                
        self.action = nn.Sequential(
            nn.Linear(
                in_features = hidden_size,
                out_features = 128),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 128,
                out_features = 2)
            )
    
    def forward(self, obs):
        return self.action(obs)



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
      
        self.value = nn.Sequential(
            nn.Linear(
                in_features = hidden_size + 2,
                out_features = 128),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = 128,
                out_features = 1)
            )
              
    def forward(self, obs, action):
        if(len(action.shape) == 1): action = action.unsqueeze(0)
        return self.value(torch.cat([obs, action.to(device)], -1))
    
    

if __name__ == "__main__":
    # Check 'em out!
    net = Summarizer((16, 16, 4))
    print(net)
    print()
    print(torch_summary(net, ((16, 16, 4),  (1,1), (1,1), (1,2))))
    
    actor = Actor()
    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, (1,hidden_size)))
    
    critic = Critic()
    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1,hidden_size),(1,2))))




# Make rtd3
@gin.configurable(module=__name__)
class RecurrentTD3(RecurrentOffPolicyRLAlgorithm):

    def __init__(
        self,
        hidden_dim=hidden_size,
        gamma=0.99,
        actor_lr=.001,
        critic_lr=.001,
        polyak=0.95, # = (1 - tau)
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2,
        max_age = 500):
        
        # hyper-parameters
      
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.polyak = polyak
      
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
      
        self.policy_delay = policy_delay
      
        # trackers
      
        self.hidden = None
        self.num_Q_updates = 0
        self.mean_Q1_value = 0
      
        # networks
      
        self.actor_summarizer = Summarizer().to(device)
        self.actor_summarizer_targ = create_target(self.actor_summarizer)
      
        self.Q1_summarizer = Summarizer().to(device)
        self.Q1_summarizer_targ = create_target(self.Q1_summarizer)
      
        self.Q2_summarizer = Summarizer().to(device)
        self.Q2_summarizer_targ = create_target(self.Q2_summarizer)
      
        self.actor = Actor().to(device)
        self.actor_targ = create_target(self.actor)
      
        self.Q1 = Critic().to(device)
        self.Q1_targ = create_target(self.Q1)
      
        self.Q2 = Critic().to(device)
        self.Q2_targ = create_target(self.Q2)
      
        # optimizers
      
        self.actor_summarizer_optimizer = Adam(self.actor_summarizer.parameters(), lr=actor_lr)
        self.Q1_summarizer_optimizer = Adam(self.Q1_summarizer.parameters(), lr=critic_lr)
        self.Q2_summarizer_optimizer = Adam(self.Q2_summarizer.parameters(), lr=critic_lr)
      
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.Q1_optimizer = Adam(self.Q1.parameters(), lr=critic_lr)
        self.Q2_optimizer = Adam(self.Q2.parameters(), lr=critic_lr)
        
        # Buffer
        
        self.episodes = RecurrentReplayBuffer(max_episode_len = max_age+1)
      

    def reinitialize_hidden(self) -> None:
        self.hidden = None
      
    def act(self, state, speed, energy, action, hc, condition = 0):
        summary, hc = self.actor_summarizer(state, speed, energy, action, hc, return_hidden=True)
        action = self.actor(summary).cpu()
        if(condition == "pin"):
            action = torch.tensor([-1,-1])
        elif(condition == "random" or random.uniform(0,1) < condition):
            action = torch.tensor([random.uniform(-1,1), random.uniform(-1,1)])
        return(action.view(2), hc)

    def update_networks(self, batch_size = 16, iterations = 1):
        if(iterations != 1): return(np.array([self.update_networks(batch_size) for _ in range(iterations)])) 
        b = self.episodes.sample(batch_size)
        if(b == False): return([None,None,None])
                
        bs, num_bptt = b.r.shape[0], b.r.shape[1]
      
        # compute summary
        
        non_action = torch.zeros((b.a.shape[0], 1, b.a.shape[2])).to(device)
        a_input = torch.cat([non_action, b.a], dim = 1)
        
        actor_summary = self.actor_summarizer(b.o, b.s, b.e, a_input)
        Q1_summary = self.Q1_summarizer(b.o, b.s, b.e, a_input)
        Q2_summary = self.Q2_summarizer(b.o, b.s, b.e, a_input)
      
        actor_summary_targ = self.actor_summarizer_targ(b.o, b.s, b.e, a_input)
        Q1_summary_targ = self.Q1_summarizer_targ(b.o, b.s, b.e, a_input)
        Q2_summary_targ = self.Q2_summarizer_targ(b.o, b.s, b.e, a_input)
        
        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary_targ[:, 1:, :]
        Q1_summary_1_T, Q1_summary_2_Tplus1 = Q1_summary[:, :-1, :], Q1_summary_targ[:, 1:, :]
        Q2_summary_1_T, Q2_summary_2_Tplus1 = Q2_summary[:, :-1, :], Q2_summary_targ[:, 1:, :]
        
        assert actor_summary.shape == (bs, num_bptt+1, self.hidden_dim)
      
        # compute predictions
      
        Q1_predictions = self.Q1(Q1_summary_1_T, b.a)
        Q2_predictions = self.Q2(Q2_summary_1_T, b.a)
      
        assert Q1_predictions.shape == (bs, num_bptt, 1)
        assert Q2_predictions.shape == (bs, num_bptt, 1)
      
        # compute targets
      
        with torch.no_grad():
            na = self.actor_targ(actor_summary_2_Tplus1)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(device)
            smoothed_na = torch.clamp(na + noise, -1, 1)
        
            n_min_Q_targ = torch.min(self.Q1_targ(Q1_summary_2_Tplus1, smoothed_na),
                                      self.Q2_targ(Q2_summary_2_Tplus1, smoothed_na))
        
            targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ
        
            assert na.shape == (bs, num_bptt, 2)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)
      
        # compute td error
      
        Q1_loss_elementwise = (Q1_predictions - targets) ** 2
        Q1_loss = mean_of_unmasked_elements(Q1_loss_elementwise, b.m)
      
        Q2_loss_elementwise = (Q2_predictions - targets) ** 2
        Q2_loss = mean_of_unmasked_elements(Q2_loss_elementwise, b.m)
      
        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()
      
        # reduce td error
      
        self.Q1_summarizer_optimizer.zero_grad()
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_summarizer_optimizer.step()
        self.Q1_optimizer.step()
      
        self.Q2_summarizer_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_summarizer_optimizer.step()
        self.Q2_optimizer.step()
      
        self.num_Q_updates += 1
      
        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3
      
            # compute policy loss
        
            a = self.actor(actor_summary_1_T)
            Q1_values = self.Q1(Q1_summary_1_T.detach(), a)  # val stands for values
            policy_loss_elementwise = - Q1_values
            policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, b.m)
        
            self.mean_Q1_value = float(-policy_loss)
            assert a.shape == (bs, num_bptt, 2)
            assert Q1_values.shape == (bs, num_bptt, 1)
            assert policy_loss.shape == ()
        
            # reduce policy loss
        
            self.actor_summarizer_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_summarizer_optimizer.step()
            self.actor_optimizer.step()
            
            policy_loss = policy_loss.cpu().detach()
        
            # update target networks
        
            polyak_update(targ_net=self.actor_summarizer_targ, pred_net=self.actor_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_summarizer_targ, pred_net=self.Q1_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_summarizer_targ, pred_net=self.Q2_summarizer, polyak=self.polyak)
        
            polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)
        else:
            policy_loss = None
      
        return(policy_loss, Q1_loss.cpu().detach(), Q2_loss.cpu().detach())
       
    def state_dict(self):
        return(
            self.actor_summarizer.state_dict(),
            self.actor_summarizer_targ.state_dict(),
            self.Q1_summarizer.state_dict(),
            self.Q1_summarizer_targ.state_dict(),
            self.Q2_summarizer.state_dict(),
            self.Q2_summarizer_targ.state_dict(),
            self.actor.state_dict(),
            self.actor_targ.state_dict(),
            self.Q1.state_dict(),
            self.Q1_targ.state_dict(),
            self.Q2.state_dict(),
            self.Q2_targ.state_dict())

    def load_state_dict(self, state_dict):
        self.actor_summarizer.load_state_dict(state_dict[0])
        self.actor_summarizer_targ.load_state_dict(state_dict[1])
        self.Q1_summarizer.load_state_dict(state_dict[2])
        self.Q1_summarizer_targ.load_state_dict(state_dict[3])
        self.Q2_summarizer.load_state_dict(state_dict[4])
        self.Q2_summarizer_targ.load_state_dict(state_dict[5])
        self.actor.load_state_dict(state_dict[6])
        self.actor_targ.load_state_dict(state_dict[7])
        self.Q1.load_state_dict(state_dict[8])
        self.Q1_targ.load_state_dict(state_dict[9])
        self.Q2.load_state_dict(state_dict[10])
        self.Q2_targ.load_state_dict(state_dict[11])
        self.episodes = RecurrentReplayBuffer()

    def eval(self):
        self.actor_summarizer.eval()
        self.actor_summarizer_targ.eval()
        self.Q1_summarizer.eval()
        self.Q1_summarizer_targ.eval()
        self.Q2_summarizer.eval()
        self.Q2_summarizer_targ.eval()
        self.actor.eval()
        self.actor_targ.eval()
        self.Q1.eval()
        self.Q1_targ.eval()
        self.Q2.eval()
        self.Q2_targ.eval()

    def train(self):
        self.actor_summarizer.train()
        self.actor_summarizer_targ.train()
        self.Q1_summarizer.train()
        self.Q1_summarizer_targ.train()
        self.Q2_summarizer.train()
        self.Q2_summarizer_targ.train()
        self.actor.train()
        self.actor_targ.train()
        self.Q1.train()
        self.Q1_targ.train()
        self.Q2.train()
        self.Q2_targ.train()
        
    def copy_networks_from(self):
        pass
    
    def load_actor(self):
        pass
    
    def save_actor(self):
        pass
