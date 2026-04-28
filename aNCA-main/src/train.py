import torch
import numpy as np
import data.Loader as Loader
import utils as utils
from NCA import NCA
from Trainer import Trainer
import os
import argparse
from Losses import FocalLoss
from torchinfo import summary
from data.DataInfo import get_num_classes
import timm 
import torchvision.models as models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from fvcore.nn import FlopCountAnalysis, parameter_count

print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_classes = get_num_classes(args.train_set)

    if_pretrained = False

    if args.predict == "resnet18":
        if if_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        model = models.resnet18(weights=weights)
        model.conv1 = torch.nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "resnet_next":
        if if_pretrained:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        else:
            weights = None
        model = models.resnext50_32x4d(weights=weights)
        model.conv1 = torch.nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "mobilenet_v2":
        if if_pretrained:
            weights = models.MobileNet_V2_Weights.DEFAULT
        else:
            weights = None
        model = models.mobilenet_v2(weights=weights)
        model.features[0][0] = torch.nn.Conv2d(args.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "squeezenet":
        if if_pretrained:
            weights = models.SqueezeNet1_1_Weights.DEFAULT
        else:
            weights = None
        model = models.squeezenet1_1(weights=weights)
        model.features[0] = torch.nn.Conv2d(args.input_channels, 64, kernel_size=3, stride=2, padding=1)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        model = model.to(device)
    elif args.predict == "shufflenet_v2":
        if if_pretrained:
            weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        else:
            weights = None
        model = models.shufflenet_v2_x0_5(weights=weights)
        model.conv1 = torch.nn.Conv2d(args.input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "efficientnet_b0":
        if if_pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        model = efficientnet_b0(weights=weights)
        model.features[0][0] = torch.nn.Conv2d(args.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "convnext_tiny":
        if if_pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None
        model = convnext_tiny(weights=weights)
        model.features[0][0] = torch.nn.Conv2d(args.input_channels, 96, kernel_size=4, stride=4, padding=0, bias=False)
        model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=num_classes)
        model = model.to(device)
    elif args.predict == "mobilevitv2":
        model = timm.create_model(
            "mobilevitv2_050",
            pretrained=if_pretrained,
            num_classes=num_classes,
            in_chans=args.input_channels
        ).to(device)
    # elif args.predict == "efficientformer":
    #     model = timm.create_model(
    #         "efficientformerv2_s0",
    #         pretrained=False,
    #         num_classes=num_classes,
    #         in_chans=args.input_channels
    #     ).to(device)
    elif args.predict == "repvgg":
        model = timm.create_model(
            "repvgg_a0",
            pretrained=if_pretrained,
            num_classes=num_classes,
            in_chans=args.input_channels
        ).to(device)
    elif args.predict == "fastvit":
        model = timm.create_model(
            "fastvit_sa12",
            pretrained=if_pretrained,
            num_classes=num_classes,
            in_chans=args.input_channels
        ).to(device)
    elif args.predict == "tinyvit":
        model = timm.create_model(
            "tiny_vit_21m_224",
            pretrained=if_pretrained,
            num_classes=num_classes,
            in_chans=args.input_channels
        ).to(device)
    else:
        model = NCA(
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

    # summary(model, input_size=(1, args.resizeW, args.resizeH, args.channel_n_1))
    # num_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameters: ", num_params)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # FLOPs
    if args.predict == "pool1" or args.predict == "aNCA" or args.predict == "hNCA":
        dummy_input_tensor = torch.randn(1, args.resizeH, args.resizeW, args.input_channels).to(device) # NCA
    else:
        dummy_input_tensor = torch.randn(1, args.input_channels, args.resizeH, args.resizeW).to(device) #baselines
    
    flops = FlopCountAnalysis(model, dummy_input_tensor)
    params = parameter_count(model)
    
    print(f"🧠 Model: {args.predict}")
    print(f"FLOPs (GFLOPs): {flops.total() / 1e9:.2f} GFLOPs")
    print(f"Parameters (M): {params[''] / 1e6:.2f} M")
    #print(f"Parameters (K): {params[''] / 1e3:.2f} K")
    #print(f"Params: {params['']:,}")

    with torch.no_grad():
        _ = model(dummy_input_tensor)

    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # in MB
    print(f"Current allocated: {allocated:.2f} MB")
    print(f"Peak allocated: {max_allocated:.2f} MB")
    # return

    train_loader, val_loader = Loader.load_datasets(args.train_set, args.fold, args.resizeW, args.resizeH, args.batch_size, args.criterion)

    if args.criterion == "Focal":
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    elif args.criterion == "CE":
        criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, device=device, args=args)    
    trainer.train(train_loader, val_loader, criterion, args.n_epochs)

    utils.log_vram_usage("after training")
    utils.evaluate_model(val_loader, args.output + "_" + args.train_set, model, num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_set', type=str, default="AML")
    parser.add_argument('--output', type=str, default="default")
    parser.add_argument('--fold', choices=[1, 2, 3, 4, 5], type=int, default=1)
    parser.add_argument('--predict', type=str, default="pool1") #"aNCA","pool1","hNCA","resnet18","mobilenet_v2","squeezenet","shufflenet_v1","shufflenet_v2"
    parser.add_argument('--criterion', type=str, default="Focal")
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--resizeW', type=int, default=64)
    parser.add_argument('--resizeH', type=int, default=64)
    parser.add_argument('--att_percent', type=float, default=0.1)
    parser.add_argument('--steps_1', type=int, default=64) 
    parser.add_argument('--channel_n_1', type=int, default=16) #128
    parser.add_argument('--hidden_size_1', type=int, default=128) 
    parser.add_argument('--steps_2', type=int, default=32)
    parser.add_argument('--channel_n_2', type=int, default=16) #128
    parser.add_argument('--hidden_size_2', type=int, default=128)
    parser.add_argument('--hidden_size_fcn', type=int, default=128) 
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=4e-4) 
    parser.add_argument('--n_epochs', type=int, default=32)
    parser.add_argument('--fire_rate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    args.output_dir = "results/" + args.output 

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # command = 'cp -r ' + 'src/ ' +  args.output_dir
    # os.system(command)

    # print(args)
    main(args)