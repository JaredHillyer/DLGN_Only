from torch.utils.tensorboard import SummaryWriter
import torch
import os
import utils as utils
import numpy as np
import time

class Trainer():
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.input_channels = args.input_channels
        self.train_set = args.train_set
        self.channel_n_1 = args.channel_n_1
        self.hidden_size_1 = args.hidden_size_1
        self.channel_n_2 = args.channel_n_2
        self.hidden_size_2 = args.hidden_size_2
        self.resizeW = args.resizeW
        self.resizeH = args.resizeH
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir
        self.predict = args.predict

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9999)

        self.writer = SummaryWriter(log_dir=self.output_dir)

    def batch_step(self, data, loss_f, train):
        inputs, targets = data
        if self.predict == "pool1" or self.predict == "aNCA" or self.predict == "hNCA":
            inputs = inputs.to(self.device)
        else:
            inputs = inputs.permute(0,3,1,2).to(self.device)
        outputs = self.model(inputs)
        # print("outputs: ", outputs)
        # print("targets: ", targets)

        loss = loss_f(outputs.to(self.device),targets.to(self.device))
        # print("loss: ", loss)
        if train == True:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return loss.detach()
    
    def train(self, train_loader, val_loader, loss_f, n_epochs):
        train_loss_total = np.zeros(n_epochs)
        val_loss_total = np.zeros(n_epochs)
        min_val_loss = float('inf')
        best_model_path = None
        
        start_time = time.time()
        print("Training...")
        for epoch in range(n_epochs):
            self.model.train()
            loss = 0
            # print("len trainloader: ", len(train_loader))
            for data in train_loader:
                loss += self.batch_step(data, loss_f, train=True)
            loss /= len(train_loader)
            loss = loss.item()
            # print("Train Loss: ", loss)
            train_loss_total[epoch] = loss
            self.writer.add_scalars('Loss', {'train': loss}, epoch)

            self.model.eval()  
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    val_loss += self.batch_step(data, loss_f, train=False)
            val_loss /= len(val_loader)
            val_loss = val_loss.item()
            # print("Val Loss: ", val_loss)
            val_loss_total[epoch] = val_loss
            self.writer.add_scalars('Loss', {'val': val_loss}, epoch)

            # if epoch == n_epochs-1:
            final_model_path = f"{self.output_dir}/Final.pt"
            torch.save(self.model.state_dict(), final_model_path)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = f"{self.output_dir}/Best.pt"
                torch.save(self.model.state_dict(), best_model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_hours = elapsed_time / 3600 
        formatted_time = f"{elapsed_hours:.2f}h"
        print(f"Training time: {formatted_time}")

        self.writer.close()

        path = self.output_dir + "/Loss.csv"
        np.savetxt(path, [train_loss_total, val_loss_total], delimiter = ',')

        output_path = f"{self.output_dir}/Loss.png"
        utils.plot_loss(train_loss_total, val_loss_total, output_path)
        
        self.model.load_state_dict(torch.load(final_model_path, map_location=self.device, weights_only=True))
        return self.model