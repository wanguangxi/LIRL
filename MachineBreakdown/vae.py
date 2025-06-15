import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
random.seed(2023)
# 定义变分编码器类
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        mean = self.fc_mean(hidden)
        log_var = self.fc_logvar(hidden)
        return mean, log_var

# 定义变分解码器类
class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VariationalDecoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output

# 定义变分自编码器类
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # 编码
        mean, log_var = self.encoder(x)
        
        # 采样
        epsilon = torch.randn_like(log_var)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        
        # 解码
        output = self.decoder(z)
        return output, mean, log_var

# 定义训练函数
def train_vae(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs in train_loader:
            # inputs, _ = data
            inputs = Variable(inputs)

            optimizer.zero_grad()

            outputs, mean, log_var = model(inputs)

            # 重构损失
            reconstruction_loss = criterion(outputs, inputs)
            # KL散度损失
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = reconstruction_loss + kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss))

# 定义测试函数
def test_vae(model, test_loader):
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    for inputs in test_loader:
        
        inputs = Variable(inputs)

        outputs, mean, log_var = model(inputs)

        # 重构损失
        reconstruction_loss = criterion(outputs, inputs)
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        loss = reconstruction_loss + kl_loss
        total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print('Test Loss: {:.4f}'.format(avg_loss))

# 创建数据集并加载数据
input_dim = 3
train_data = torch.randn(1000, input_dim)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_data = torch.randn(100, input_dim)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# 定义模型并进行训练和测试
hidden_dim = 256
latent_dim = 5

vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)

num_epochs = 1000
train_vae(vae, train_loader, num_epochs)
test_vae(vae, test_loader)

# 使用模型进行输入重建
input_tensor = torch.tensor([[0.5, 1.0, -0.5]])
output_tensor, _, _ = vae(input_tensor)
print('Input:', input_tensor)
print('Output:', output_tensor)
