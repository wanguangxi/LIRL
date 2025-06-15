import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import env as ENV
import numpy as np
# random.seed(20230911)

#loss

history_loss = []
# # 定义自动编码器类
class Conditional_Autoencoder(nn.Module):
    def __init__(self, input_dim, state_dims,hidden_dims):
        super(Conditional_Autoencoder, self).__init__()

        encoder_layers = []
        prev_dim = input_dim+state_dims

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=False))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)
        

        decoder_layers = []
        prev_dim = prev_dim+state_dims
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU(inplace=False))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.ReLU(inplace=False))

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self,x,s):
        inputs = torch.cat([x, s], -1) 
        encoded = self.encoder(inputs)

        latent = torch.cat([encoded, s], -1) 
        decoded = self.decoder(latent)
        extra_layer = nn.Sigmoid()
        decoded[-1]= extra_layer(decoded[-1]) 
        return decoded
# 定义训练函数
def train_autoencoder(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        print("Training epoch:",epoch)
        for s,a in zip(train_loader.dataset[0],train_loader.dataset[1]):

            action = Variable(a)
            optimizer.zero_grad()
            outputs = model(action,s)
            loss = criterion(outputs, action)
            # print(loss)
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss = total_loss+loss.item()
        history_loss.append(total_loss)
        
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss))

buffer_limit = 1000000
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
        
        
           
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)      
        s_lst, a_lst = [], []

        for transition in mini_batch:
            s, a = transition
            s_lst.append(s)
            a_lst.append(a)
            
        # print(s_lst.size())    
        s_lst_= torch.tensor(np.array(s_lst)).to(torch.float32)
        a_lst_= torch.tensor(np.array(a_lst)).to(torch.float32)
        
        return s_lst_,a_lst_
    
    def size(self):
        return len(self.buffer)



def action_mask(env,Job_data,Machine_data,Param_data):
    origin_action = [Job_data,Machine_data,Param_data]
    valid_flag = True
    valid_job =[]
    valid_robot = []
    task_set = env.task_set
    for job_id in range(len(task_set)):
        job = task_set[job_id]
        finished = False
        for op in range(len(task_set[job_id])):
            task = job[op]
            if task.state:
                finished = True
            else:
                finished = False
                break
        if not finished:
            valid_job.append(job_id)

    for robot_id in range(len(env.robot_state)):
        if env.robot_state[robot_id]==1:
            valid_robot.append(robot_id)
    
    if not Job_data in valid_job:
            Job_data = random.choice(valid_job)
    if not Machine_data in valid_robot:
            Machine_data = random.choice(valid_robot)
    action = [Job_data,Machine_data,Param_data]
    if origin_action!= action:
        valid_flag = False
    return valid_flag,origin_action,action


if __name__ == '__main__':
    # 创建数据集并加载数据
    print("训练作业数为10的编码器")
    num_of_jobs = 10
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 5000
    score_records = []
    
    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    state_size= len(env.state)
    action_size = len(env.action)
    #随机策略采样
    memory = ReplayBuffer()
    for n_epi in range(episodes):
        if n_epi%100 ==0:
            print("training episode:",n_epi)
        s = env.reset()
        done = False
        while not done:
            Operation_data = random.randint(0, num_of_jobs-1)
            Machine_data = random.randint(0, num_of_robots-1)
            Param_data = random.uniform(0, 1)
            
            action = action_mask(env, Operation_data,Machine_data,Param_data)
            s_prime, r, done= env.step(action)
            memory.put((s[0:num_of_jobs*5+num_of_robots],action))
            s = s_prime

                    

    train_data = memory.sample(memory.size())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 定义模型并进行训练
    hidden_dims = [16,64,8]
    state_dims=num_of_jobs*5+num_of_robots
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    num_epochs = 50  
    train_autoencoder(autoencoder, train_loader, num_epochs)
    # 存储模型
    torch.save(autoencoder.state_dict(), "conditional_autoencoder_10.pth")

    # 加载模型
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    autoencoder.load_state_dict(torch.load("conditional_autoencoder_10.pth"))
    # 使用模型进行输入重建
    input_tensor = torch.tensor([4, 1, 0.56]).to(torch.float32)
    state =  torch.tensor(env.reset()).to(torch.float32)

    inputs = torch.cat([input_tensor, state[0:num_of_jobs*5+num_of_robots]], -1) 
    encode = autoencoder.encoder(inputs)
    print("latent:",encode)
    output_tensor = autoencoder(input_tensor,state[0:num_of_jobs*5+num_of_robots])
    print('Input:', input_tensor)
    print('Output:', output_tensor)




# 创建数据集并加载数据
    print("训练作业数为20的编码器")
    num_of_jobs = 20
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 5000
    score_records = []
    
    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    state_size= len(env.state)
    action_size = len(env.action)
    #随机策略采样
    memory = ReplayBuffer()
    for n_epi in range(episodes):
        if n_epi%100 ==0:
            print("training episode:",n_epi)
        s = env.reset()
        done = False
        while not done:
            Operation_data = random.randint(0, num_of_jobs-1)
            Machine_data = random.randint(0, num_of_robots-1)
            Param_data = random.uniform(0, 1)
            
            action = action_mask(env, Operation_data,Machine_data,Param_data)
            s_prime, r, done= env.step(action)
            memory.put((s[0:num_of_jobs*5+num_of_robots],action))
            s = s_prime

                    

    train_data = memory.sample(memory.size())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 定义模型并进行训练
    hidden_dims = [16,64,8]
    state_dims=num_of_jobs*5+num_of_robots
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    num_epochs = 50  
    train_autoencoder(autoencoder, train_loader, num_epochs)
    # 存储模型
    torch.save(autoencoder.state_dict(), "conditional_autoencoder_20.pth")

    # 加载模型
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    autoencoder.load_state_dict(torch.load("conditional_autoencoder_20.pth"))
    # 使用模型进行输入重建
    input_tensor = torch.tensor([4, 1, 0.56]).to(torch.float32)
    state =  torch.tensor(env.reset()).to(torch.float32)

    inputs = torch.cat([input_tensor, state[0:num_of_jobs*5+num_of_robots]], -1) 
    encode = autoencoder.encoder(inputs)
    print("latent:",encode)
    output_tensor = autoencoder(input_tensor,state[0:num_of_jobs*5+num_of_robots])
    print('Input:', input_tensor)
    print('Output:', output_tensor)



# 创建数据集并加载数据
    print("训练作业数为50的编码器")
    num_of_jobs = 50
    num_of_robots = 3
    alpha = 0.2
    beta = 0.8
    episodes = 5000
    score_records = []
    
    env = ENV.Env(num_of_jobs, num_of_robots,alpha,beta)
    state_size= len(env.state)
    action_size = len(env.action)
    #随机策略采样
    memory = ReplayBuffer()
    for n_epi in range(episodes):
        if n_epi%100 ==0:
            print("training episode:",n_epi)
        s = env.reset()
        done = False
        while not done:
            Operation_data = random.randint(0, num_of_jobs-1)
            Machine_data = random.randint(0, num_of_robots-1)
            Param_data = random.uniform(0, 1)
            
            action = action_mask(env, Operation_data,Machine_data,Param_data)
            s_prime, r, done= env.step(action)
            memory.put((s[0:num_of_jobs*5+num_of_robots],action))
            s = s_prime

                    

    train_data = memory.sample(memory.size())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 定义模型并进行训练
    hidden_dims = [16,64,8]
    state_dims=num_of_jobs*5+num_of_robots
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    num_epochs = 50  
    train_autoencoder(autoencoder, train_loader, num_epochs)
    # 存储模型
    torch.save(autoencoder.state_dict(), "conditional_autoencoder_50.pth")

    # 加载模型
    autoencoder = Conditional_Autoencoder(action_size,state_dims,hidden_dims)
    autoencoder.load_state_dict(torch.load("conditional_autoencoder_50.pth"))
    # 使用模型进行输入重建
    input_tensor = torch.tensor([4, 1, 0.56]).to(torch.float32)
    state =  torch.tensor(env.reset()).to(torch.float32)

    inputs = torch.cat([input_tensor, state[0:num_of_jobs*5+num_of_robots]], -1) 
    encode = autoencoder.encoder(inputs)
    print("latent:",encode)
    output_tensor = autoencoder(input_tensor,state[0:num_of_jobs*5+num_of_robots])
    print('Input:', input_tensor)
    print('Output:', output_tensor)











    # import matplotlib.pyplot as plt
    # x = range(len(history_loss))
    # y =  history_loss
    # plt.plot(x,y)
    # plt.show()