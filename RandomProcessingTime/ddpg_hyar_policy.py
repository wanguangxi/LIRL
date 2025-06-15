import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env as ENV
import ae as AE
import human_policy as hp
import pickle
# random.seed(20230826)
#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.98
batch_size   = 256
buffer_limit = 1000000
tau          = 0.005 # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
        
        
           
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        # print(s_lst.size())    
        s_lst_= torch.FloatTensor(s_lst)
        a_lst_= torch.tensor(a_lst, dtype=torch.float)
        r_lst_= torch.tensor(r_lst, dtype=torch.float)
        s_prime_lst_ = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask_lst_ = torch.tensor(done_mask_lst, dtype=torch.float)

        return s_lst_,a_lst_,r_lst_,s_prime_lst_,done_mask_lst_
    
    def size(self):
        return len(self.buffer)
    
outlayer = nn.Sigmoid()
class MuNet(nn.Module):
    def __init__(self,state_size, action_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = outlayer(self.fc_mu(x)) 
        return mu

class QNet(nn.Module):
    def __init__(self,state_size, action_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, 64)
        self.fc_a = nn.Linear(action_size,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.05, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    

    target =torch.unsqueeze(r, dim=1)+ gamma * q_target(s_prime, mu_target(s_prime)).mul(done_mask) 
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main(num_of_jobs, num_of_robots,alpha,beta,episodes):

    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    state_size= len(env.state)
    action_size = len(env.action)
    
    hidden_dims = [16,64,8]
    autoencoder = AE.Autoencoder(action_size, hidden_dims)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    latent_action = hidden_dims[-1]
    

    memory = ReplayBuffer()
    with open("human.pkl", "rb") as file:
        memory = pickle.load(file)

    q, q_target = QNet(state_size,latent_action), QNet(state_size,latent_action)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_size,latent_action), MuNet(state_size,latent_action)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(latent_action))

    
    
  
    print("memory.size():",memory.size())  
    if memory.size()>500:
        for i in range(100):
            # print("Training time:",i)
            train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
            soft_update(mu, mu_target)
            soft_update(q,  q_target)

    for n_epi in range(episodes):
        s = env.reset()
        # print(s)
        done = False
        # print("episode:",n_epi)
        steps = 0
        
        while not done:
            a = mu(torch.from_numpy(s).float()) 
            # print("latent_action:",a)
            a = a + torch.from_numpy(ou_noise())
            a = torch.clamp(a, 0, 1)
            a=a.to(torch.float32)
            action = autoencoder.decoder(a)
            extra_layer = nn.Sigmoid()
            action[-1]= extra_layer(action[-1]) 

            s_prime, r, done= env.step(action.detach().numpy())
            memory.put((s,a.detach().numpy(),r,s_prime,done))
            score += r
            s = s_prime
            steps += 1
            
          
        if memory.size()>500:
            for i in range(20):
                # print("Training time:",i)
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score_record.append(score/print_interval)
            score = 0.0

    
    torch.save(mu.state_dict(), "mu_model.pth")
    torch.save(mu_target.state_dict(), "mu_target_model.pth")
    torch.save(q.state_dict(), "q_model.pth")
    torch.save(q_target.state_dict(), "q_target_model.pth")


if __name__ == '__main__':
    num_of_jobs = 5
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 10000
    score_record = []
    main(num_of_jobs, num_of_robots,alpha,beta,episodes)

    import matplotlib.pyplot as plt
    x = range(len(score_record))
    y =  score_record
    plt.plot(x,y)
    plt.show()