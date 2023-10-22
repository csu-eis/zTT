import torch
from torch import nn
import numpy as np
from numpy import random
import time
from collections import namedtuple
import os
import pickle

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
        
class DQN_AB(nn.Module):
    def __init__(self, s_dim=10, h_dim=25, branches=[1,2,3]):
        super(DQN_AB, self).__init__()
        self.s_dim, self.h_dim = s_dim, h_dim
        self.branches = branches
        self.shared = nn.Sequential(nn.Linear(self.s_dim, self.h_dim), nn.ReLU(),)
        self.shared_state = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim), 
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim*2), 
            nn.ReLU())
        self.domains, self.outputs = [], []
        for i in range(len(branches)):
            layer = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim*2), 
                nn.ReLU(),
                nn.Linear(self.h_dim*2, self.h_dim*2), 
                nn.ReLU()
                )
            self.domains.append(layer)
            layer_out = nn.Sequential(
                nn.Linear(self.h_dim*4, self.h_dim*2),
                nn.ReLU(),
            	nn.Linear(self.h_dim*2, branches[i]), 
                )
            self.outputs.append(layer_out)
        self.layer_norm = nn.LayerNorm(self.h_dim*4)
        
    def forward(self, x):
        if len(x.shape)==1:
            x= x.view(-1,self.s_dim)
        # return list of tensors, each element is Q-Values of a domain
        f = self.shared(x)
        s = self.shared_state(f)
        outputs = []
        for i in range(len(self.branches)):
            branch = self.domains[i](f)
            branch = torch.cat([branch,s],dim=1)
            out = self.layer_norm(branch)
            out = self.outputs[i](out)
            # out = nn.functional.softmax(self.outputs[i](branch),dim=1)
            outputs.append(out)
        return outputs
    



def save_checkpoint(state, savepath, flag=True):
    """Save for general purpose (e.g., resume training)"""
    if not os.path.isdir(savepath):
        os.makedirs(savepath, 0o777)
    # timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    torch.save(state, filename)


def load_checkpoint(savepath, flag=True):
    """Load for general purpose (e.g., resume training)"""
    if flag:
        filename = os.path.join(savepath, "best_ckpt.pth.tar")
    else:
        filename = os.path.join(savepath, "newest_ckpt.pth.tar")
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

    
    
class DQN_AGENT_AB():
	def __init__(self, s_dim, h_dim, branches, buffer_size, params):
		"""
		s_dim是输入向量的维度
		h_dim是隐藏层的温度
		branches是一个列表内的每一个数字代表该分支的Actions大小。
		"""
		torch.manual_seed(202310)
		self.eps = 0.8
		# self.params = params
		# 2D action space
		self.actions = [np.arange(i) for i in branches]
		# Experience Replay(requires belief state and observations)
		self.mem = ReplayMemory(buffer_size)
		# Initi networks
		self.evaluate_net = DQN_AB(s_dim, h_dim, branches)
		self.target_net = DQN_AB(s_dim, h_dim, branches)
		
		
		self.target_net.load_state_dict(self.evaluate_net.state_dict())
		self.target_net.eval()

		self.optimizer = torch.optim.RMSprop(self.evaluate_net.parameters())
		self.criterion = nn.SmoothL1Loss() # Huber loss
	
	
	def max_action(self, state):
		# actions for multidomains
		max_actions = []
		with torch.no_grad():
			# Inference using policy_net given (domain, batch, dim)
			q_values = self.evaluate_net(state)
			for i in range(len(q_values)):
				domain = q_values[i].max(dim=1).indices
				max_actions.append(self.actions[i][domain])
		return max_actions

	def e_gready_action(self, actions, eps):
		# Epsilon-Gready for exploration
		final_actions = []
		for i in range(len(actions)):
			p = np.random.random()
			if isinstance(actions[i],np.ndarray):
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					# randint in (0, domain_num), for batchsize
					final_actions.append(np.random.randint(len(self.actions[i]),size=len(actions[i])))
			else:
				if p < 1- eps:
					final_actions.append(actions[i])
				else:
					final_actions.append(np.random.choice(self.actions[i]))
		final_actions = [int(i) for i in final_actions]
		return final_actions

	def select_action(self, state):
		return self.e_gready_action(self.max_action(state),self.eps)

	def train(self, n_round, n_update, n_batch):
		# Train on policy_net
		losses = []
		self.target_net.eval()
		self.evaluate_net.train()
		train_loader = torch.utils.data.DataLoader(self.mem, shuffle=True, batch_size=n_batch)
		
		GAMMA = 1.0
		ALPHA = 0.3
	
		# Calcuate loss for each branch and then simply sum up
		for i, trans in enumerate(train_loader):
			loss = 0.0 # initialize loss at the beginning of each batch
			states, actions, next_states, rewards = trans
			with torch.no_grad():
				next_state_q_result = self.target_net(next_states)

			curr_state_q_result = self.evaluate_net(states)
   
   
			# 这里计算LOSS
			for j in range(len(self.actions)):
                # 计算next state的Q_max
				next_state_action_q_value = next_state_q_result[j].max(dim=1)[0].detach()
				expected_state_action_q_value = (next_state_action_q_value*GAMMA) + rewards.float()
				expected_state_action_q_value *= ALPHA
		
				branch_actions = actions[j].long()
				curr_state_action_q_value = curr_state_q_result[j].gather(1, branch_actions.unsqueeze(1))
				loss += self.criterion(curr_state_action_q_value, expected_state_action_q_value.unsqueeze(1))
			losses.append(loss.item())
			self.optimizer.zero_grad()
			loss.backward()


			self.optimizer.step()
		return losses

	def save_model(self, n_round, savepath):
		save_checkpoint({'epoch': n_round, 'model_state_dict':self.target_net.state_dict(),
	        'optimizer_state_dict':self.optimizer.state_dict()}, savepath)
		f = open(os.path.join(savepath,"memory"), 'wb')
		pickle.dump(self.mem,f)
		f.close()

	def load_model(self, loadpath):
		if not os.path.isdir(loadpath): os.makedirs(loadpath)
		checkpoint =load_checkpoint(loadpath)
		if checkpoint is not None:
			self.evaluate_net.load_state_dict(checkpoint['model_state_dict'])
			self.target_net.load_state_dict(checkpoint['model_state_dict'])
			self.target_net.eval()
		if os.path.exists(os.path.join(loadpath,"memory")):
			f = open(os.path.join(loadpath,"memory"),'rb')
			self.mem = pickle.load(f)
			f.close()

	def sync_model(self):
		self.target_net.load_state_dict(self.evaluate_net.state_dict())
  