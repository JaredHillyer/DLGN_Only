import torch
import numpy as np
import data.Loader as Loader
import utils as utils
from src.hNCA import hNCA
from Trainer import Trainer
import os
import argparse
from Losses import FocalLoss
from data.DataInfo import get_num_classes

torch.cuda.empty_cache()
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.reset_peak_memory_stats()

    num_classes = get_num_classes(args.train_set)

    model = hNCA(
        input_channels = args.input_channels,
        resizeW = args.resizeW,
        resizeH = args.resizeH,
        hidden_size_fcn = args.hidden_size_fcn,
        predict_head = args.predict,
        num_classes = num_classes,
        channel_n_1 = args.channel_n_1, 
        hidden_size_1 = args.hidden_size_1, 
        steps_1 = args.steps_1,
        channel_n_2 = args.channel_n_2, 
        hidden_size_2 = args.hidden_size_2,
        steps_2 = args.steps_2,
        att_percent=args.att_percent,
        device = device, 
        dropout = args.dropout,
        fire_rate = args.fire_rate).to(device)
    
    # from torchinfo import summary
    # summary(model, input_size=(1, args.resizeW, args.resizeH, args.channel_n_1))
    # num_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameters: ", num_params)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    train_loader, val_loader = Loader.load_datasets(args.train_set, args.fold, args.resizeW, args.resizeH, args.batch_size, args.criterion)

    if args.criterion == "Focal":
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    elif args.criterion == "CE":
        criterion = torch.nn.CrossEntropyLoss()

    if args.mode == "train":
        trainer = Trainer(model, device=device, args=args)    
        trainer.train(train_loader, val_loader, criterion, args.n_epochs)
        utils.evaluate_model(val_loader, args.output + "_" + args.train_set, model, num_classes)
    elif args.mode == "eval":
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model loaded on {next(model.parameters()).device}")
        utils.evaluate_model(val_loader, args.output + "_" + args.train_set, model, num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--criterion', type=str, default="Focal")
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--resizeW', type=int, default=64)
    parser.add_argument('--resizeH', type=int, default=64)
    parser.add_argument('--steps_1', type=int, default=32)
    parser.add_argument('--channel_n_1', type=int, default=128)
    parser.add_argument('--hidden_size_1', type=int, default=128)
    parser.add_argument('--steps_2', type=int, default=16)
    parser.add_argument('--channel_n_2', type=int, default=128)
    parser.add_argument('--hidden_size_2', type=int, default=128)
    parser.add_argument('--hidden_size_fcn', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4) 
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--fire_rate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mode', type=str, default="eval")
    parser.add_argument('--model_path', type=str, default="results/AML+1/Final.pt")


    args = parser.parse_args()
    args.output_dir = "results/" + args.output 

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)