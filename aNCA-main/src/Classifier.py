# write a fc clasiffier, input(b, h, w, c)  output(b, num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_Classifier(nn.Module):
    def __init__(self, 
                 num_classes,
                 channel_n_1,
                 hidden_size_fcn,
                 att_percent,
                 resizeW,
                 resizeH):
        super(FC_Classifier, self).__init__()
        self.fc2 = nn.Linear(channel_n_1, hidden_size_fcn)
        self.fc3 = nn.Linear(hidden_size_fcn, num_classes)
        self.att_percent = att_percent
        self.resizeW = resizeW
        self.resizeH = resizeH

    def forward(self, x):
    # ([b, c, 64, 64]) -> (b, 13)
        # x = x * self.mask
        x = x.view(x.size(0), x.size(1), -1) #(b, c, 64*64)
        x, _ = torch.sort(x, dim=2, descending=True) #(b, c, 64*64)
        max = x[:,:,:int(self.att_percent * self.resizeW * self.resizeH)].mean(dim=2) #(b, c)

        out = self.fc2(max) #(b, hidden_size_fcn)
        out = F.relu(out)
        out = self.fc3(out) #(b, num_classes)
        return out
    
    def train(self, data, target, epochs, optimizer, criterion, path):
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(data)
            # #softmax output
            # output = F.softmax(output, dim=1)
            # print(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, epoch, epochs,100. * epoch / epochs, loss.item()))
            torch.save(self.state_dict(), path)        

