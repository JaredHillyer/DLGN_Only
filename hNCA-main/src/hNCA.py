import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils

class hNCA(nn.Module):
    def __init__(self, 
                input_channels,
                resizeW,
                resizeH,
                hidden_size_fcn,
                predict_head,
                num_classes,
                channel_n_1, 
                hidden_size_1, 
                steps_1,
                channel_n_2, 
                hidden_size_2,
                steps_2,
                att_percent,
                device, 
                dropout,
                fire_rate):
        super(hNCA, self).__init__()

        self.nca1 = NCA_backbone(input_channels, channel_n_1, hidden_size_1, device, dropout, fire_rate, steps_1)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.fire_rate = fire_rate
        self.input_channels = input_channels
        self.predict_head = predict_head

        self.steps_1 = steps_1
        self.channel_n_1 = channel_n_1

        self.steps_2 = steps_2
        self.channel_n_2 = channel_n_2
        self.hidden_size_fcn = hidden_size_fcn

        self.resizeW = resizeW
        self.resizeH = resizeH
        self.att_percent = att_percent

        self.nca2 = NCA_backbone(input_channels, channel_n_2, hidden_size_2, device, dropout, fire_rate, steps_2)
        self.fc2 = nn.Linear(channel_n_2, hidden_size_fcn)
        self.fc3 = nn.Linear(hidden_size_fcn, num_classes)
        
        self.to(device)

    def forward(self, x):
        x = utils.make_seed(x, self.channel_n_1, self.device)
        x = self.nca1(x)  # (b, 64, 64, c1)
        out = self.classify(x)
        return out 
    
    def classify(self, x): # ([b, 64, 64, c1]) -> (b, 13)
        x = x.permute(0, 3, 1, 2) # ([b, c1, 64, 64])

        x = F.adaptive_max_pool2d(x, (16, 16)) # (b, c1, 16, 16)
        x = x.transpose(1,3) #(b, 16, 16, c1)
        if(self.channel_n_2 > self.channel_n_1):
            x = utils.make_seed(x, self.channel_n_2, self.device)
        elif(self.channel_n_2 < self.channel_n_1):
            raise ValueError("channel_n_2 should be greater equal than channel_n_1")
        x = self.nca2(x) #(b, 16, 16, c2)
        x = x.transpose(1,3) #(b, c2, 16, 16)
        max = F.adaptive_max_pool2d(x, (1, 1)) #(b, c2, 1, 1)
        max = max.view(max.size(0), -1)

        out = self.fc2(max) #(b, hidden_size_fcn)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out) #(b, num_classes)

        return out

class NCA_backbone(nn.Module):
    def __init__(self, 
                input_channels,
                channel_n, 
                hidden_size, 
                device, 
                dropout,
                fire_rate,
                steps):
                
        super(NCA_backbone, self).__init__()

        self.input_channels = input_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, groups = channel_n, padding_mode="reflect")

        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, channel_n)
        self.to(device)

    def perceive(self, x):
        y = torch.cat((x, self.p0(x),self.p1(x)),1) # ([b, 3c, 64, 64])
        return y

    def update(self, x_in): # ([b, 64, 64, c])
        x = x_in.transpose(1,3) # ([b, c, 64, 64])

        dx = self.perceive(x) 
        dx = dx.transpose(1,3) # ([b, 64, 64, 3c])

        dx = F.relu(self.fc0(dx))

        dx = self.dropout(dx)
        dx = self.fc1(dx) # ([b, 64, 64, c])

        stochastic = torch.rand_like(dx[..., :1]) >= self.fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic 

        x = x + dx.transpose(1,3)
        x = x.transpose(1,3) # ([b, 64, 64, c])
        return x

    def forward(self, x):
        for _ in range(self.steps):
            x2 = self.update(x).clone()
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
        return x # ([b, 64, 64, c])

