import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils

class NCA(nn.Module):
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
        super(NCA, self).__init__()

        self.nca1 = NCA_backbone(input_channels, channel_n_1, hidden_size_1, device, dropout, fire_rate, steps_1)
        # self.dropout = nn.Dropout(dropout)
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

        if predict_head == "pool1":
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)

        elif predict_head in ["hNCA", "avgfusion"]:
            self.nca2 = NCA_backbone(input_channels, channel_n_2, hidden_size_2, device, dropout, fire_rate, steps_2)
            self.fc2 = nn.Linear(channel_n_2, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)

        elif predict_head == "aNCA":
            self.attention = nn.Parameter(torch.randn(resizeH, resizeW))
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)
        
        elif predict_head == "attention_lastchannel":
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)

        elif predict_head == "attention_conv":
            self.conv1x1 = nn.Conv2d(channel_n_1, 1, kernel_size=1, stride=1, padding=0) #(b, 128, 64, 64) -> (b, 1, 64, 64)
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)

        elif predict_head == "mic":
            self.conv1x1 = nn.Conv2d(channel_n_1, 1, kernel_size=1, stride=1, padding=0) #(b, 128, 16, 16) -> (b, 1, 16, 16)
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)
        
        elif predict_head == "patch_attention" or predict_head == "patch_aNCA":
            self.attention = nn.Parameter(torch.randn(resizeW, resizeH))

            self.patch_num = 16
            self.patch_sizeX = self.resizeW // self.patch_num
            self.patch_sizeY = self.resizeH // self.patch_num
            self.proj = nn.Conv2d(channel_n_1, channel_n_1, kernel_size=self.patch_size, stride=self.patch_size) #(b, 128, 64, 64) -> (b, 128, 16, 16)
            
            self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)

        elif predict_head == "embed_patch":
            # self.attention = nn.Parameter(torch.randn(resizeW, resizeH))

            self.patch_num = 16
            self.patch_sizeX = self.resizeW // self.patch_num
            self.patch_sizeY = self.resizeH // self.patch_num
            self.embed_dim = 1
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_num * self.patch_num, self.embed_dim))
            self.proj = nn.Conv2d(channel_n_1, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size) #(b, 128, 64, 64) -> (b, d, 16, 16)
            
            self.fc2 = nn.Linear(self.patch_num * self.patch_num, hidden_size_fcn)
            self.fc3 = nn.Linear(hidden_size_fcn, num_classes)
        
        self.to(device)

    def forward(self, x):
        x = utils.make_seed(x, self.channel_n_1, self.device)
        # print(f"Memory 0: {utils.get_memory_usage():.2f} MB")
        x = self.nca1(x)  # (b, 64, 64, c1)
        # print(f"Memory 1: {utils.get_memory_usage():.2f} MB")
        out = self.classify(x)
        # print(f"Memory 2: {utils.get_memory_usage():.2f} MB")

        # x1 = x.clone().detach().cpu()
        return out #, x1
    
    def classify(self, x):
        # ([b, 64, 64, c1]) -> (b, 13)
        x = x.permute(0, 3, 1, 2) # ([b, c1, 64, 64])

        if self.predict_head == "pool1":
            max = F.adaptive_max_pool2d(x, (1, 1)) #(b, c1, 1, 1)
            max = max.view(max.size(0), -1)

        elif self.predict_head == "hNCA":
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

        # elif self.predict_head == "avgfusion":
        #     x = F.adaptive_avg_pool2d(x, (16, 16)) # (b, c, 16, 16)
        #     x_before = x.clone() #(b, c, 16, 16)
        #     x = x.transpose(1,3) #(b, 16, 16, c)
        #     if(self.channel_n_2 != self.channel_n_1):
        #         raise ValueError("channel_n_2 should be equal to channel_n_1")
        #     x = self.nca2(x) #(b, 16, 16, c)
        #     x_after = x.transpose(1,3).clone() #(b, c, 16, 16)
        #     x = x_before + x_after
        #     max = F.adaptive_avg_pool2d(x, (1, 1)) #(b, c, 1, 1)
        #     max = max.view(max.size(0), -1) #(b, c)

        elif self.predict_head == "aNCA":
            x = x * F.sigmoid(self.attention)
            x = x.view(x.size(0), x.size(1), -1) #(b, c, 64*64)
            x, _ = torch.sort(x, dim=2, descending=True) #(b, c, 64*64)
            max = x[:,:,:int(self.att_percent * self.resizeW * self.resizeH)].mean(dim=2) #(b, c)

        elif self.predict_head == "patch_attention":
            x = x * F.sigmoid(self.attention) #(b, c, 64, 64)
            x = self.proj(x) #(b, c, 16, 16)
            max = F.adaptive_max_pool2d(x, (1, 1)) #(b, c, 1, 1)
            max = max.view(max.size(0), -1) # (b, c)
        elif self.predict_head == "patch_aNCA":
            x = x * F.sigmoid(self.attention) #(b, c, 64, 64)
            x = self.proj(x) #(b, c, 16, 16)
            x = x.view(x.size(0), x.size(1), -1) #(b, c, 16*16)
            x, _ = torch.sort(x, dim=2, descending=True) #(b, c, 16*16)
            max = x[:,:,:int(self.att_percent * self.resizeW * self.resizeH)].mean(dim=2) #(b, c)

        elif self.predict_head == "attention_lastchannel":
            attention = x[:,-1,:,:].unsqueeze(1) #(b, 1, 64, 64)
            x = x * F.sigmoid(attention)
            max = F.adaptive_max_pool2d(x, (1, 1)) #(b, c, 1, 1)
            max = max.view(max.size(0), -1) # (b, c)
        elif self.predict_head == "attention_conv":
            attention = self.conv1x1(x) #(b, 1, 64, 64)
            x = x * F.sigmoid(attention)
            max = F.adaptive_max_pool2d(x, (1, 1)) #(b, c, 1, 1)
            max = max.view(max.size(0), -1)
        elif self.predict_head == "mic":
            x_down = F.avg_pool2d(x, kernel_size=4)  # (B, C, 16, 16)
            attention = self.conv1x1(x_down)  # (B, 1, 16, 16)
            attention = F.interpolate(attention, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)  # (B, 1, 64, 64)
            B, C, H, W = attention.shape
            attention = attention.contiguous().view(B, C, -1)  # (B, 1, 4096)
            attention = F.softmax(attention, dim=-1)
            attention = attention.view(B, C, H, W)
            # Weighted sum over spatial dimensions
            x = x * attention # (B, C, 64, 64)
            max = x.sum([2,3]) # (B, C)

        elif self.predict_head == "embed_patch":
            # (b, c, 64, 64)
            # x = x * F.sigmoid(self.attention)
            embed = self.proj(x) #(b, d, 16, 16)
            embed = embed.view(embed.size(0), embed.size(1), -1) #(b, d, 16*16)
            embed = embed.transpose(1,2) #(b, 16*16, d)
            embed = embed + self.pos_embed
            max = embed.view(embed.size(0), -1) #(b, 16*16*d)

        else:
            raise ValueError("predict_head not recognized, you have {}".format(self.predict_head))
        
        out = self.fc2(max) #(b, hidden_size_fcn)
        out = F.relu(out)
        # out = self.dropout(out)
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
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, channel_n)
        self.to(device)

    def perceive(self, x):
        # z1 = self.p0(x) 
        # print(f"Memory a: {utils.get_memory_usage():.2f} MB")
        # z2 = self.p1(x)
        # print(f"Memory b: {utils.get_memory_usage():.2f} MB")
        # y = torch.cat((x,z1,z2),1) # ([b, 3c, 64, 64])
        # print(f"Memory c: {utils.get_memory_usage():.2f} MB")
        y = torch.cat((x, self.p0(x),self.p1(x)),1) # ([b, 3c, 64, 64])
        # print(f"Memory c: {utils.get_memory_usage():.2f} MB")
        return y

    def update(self, x_in): # ([b, 64, 64, c])
        x = x_in.transpose(1,3) # ([b, c, 64, 64])

        # torch.cuda.synchronize()
        # utils.log_vram_usage("0")

        dx = self.perceive(x) 
        dx = dx.transpose(1,3) # ([b, 64, 64, 3c])

        # torch.cuda.synchronize()
        # utils.log_vram_usage("1")

        dx = F.relu(self.fc0(dx), inplace=True)

        # print(f"Memory e: {utils.get_memory_usage():.2f} MB")

        # torch.cuda.synchronize()
        # utils.log_vram_usage("2")

        # dx = self.dropout(dx)
        dx = self.fc1(dx) # ([b, 64, 64, c])

        # stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1]) >= self.fire_rate
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

