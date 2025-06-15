import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
random.seed(20230821)

#loss

history_loss = []
# # 定义自动编码器类
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)
        

        decoder_layers = []

        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        x = x.to(torch.float32)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        extra_layer = nn.Sigmoid()
        decoded[:,-1]= extra_layer(decoded[:,-1]) 
        return decoded
# 定义训练函数
def train_autoencoder(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for input in train_loader:
            inputs = Variable(input)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        history_loss.append(total_loss)
        
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss))



if __name__ == '__main__':
    # 创建数据集并加载数据
    input_dim = 3
    data_num = 10000
    numb_of_job = 5
    operation_num = 5
    Operation_data = torch.randint(0,numb_of_job, (data_num,1))
    Machine_data = torch.randint(0,3, (data_num,1))
    Param_data = torch.torch.empty(data_num,1 ).uniform_(0, 1)
    train_data = torch.cat((Operation_data, Machine_data), dim=-1)
    train_data = torch.cat((train_data, Param_data), dim=-1)
    
   
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 定义模型并进行训练
    hidden_dims = [16,64,8]

    autoencoder = Autoencoder(input_dim, hidden_dims)

    num_epochs = 1000  
    print("training...")
    train_autoencoder(autoencoder, train_loader, num_epochs)

    # 存储模型
    torch.save(autoencoder.state_dict(), "autoencoder.pth")

    # 加载模型
    autoencoder = Autoencoder(input_dim, hidden_dims)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    # 使用模型进行输入重建
    input_tensor = torch.tensor([[4, 1, 0.56]])
    encode = autoencoder.encoder(input_tensor)
    print("latent:",encode)
    output_tensor = autoencoder(input_tensor)
    print('Input:', input_tensor)
    print('Output:', output_tensor)

    import matplotlib.pyplot as plt
    x = range(len(history_loss))
    y =  history_loss
    plt.plot(x,y)
    plt.show()