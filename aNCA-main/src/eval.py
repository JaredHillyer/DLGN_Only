import matplotlib.pyplot as plt
import utils as utils
import time
from torchvision import transforms
from torchinfo import summary
from PIL import Image
from utils import Adder
from NCA import NCA
import argparse
import torch.nn.functional as F
import numpy as np
import data.Loader as Loader
from data.GetData import get_test_sample
from data.DataInfo import get_num_classes
import torchvision.models as models
import os
import torch
import multiprocessing
import psutil

# python -m torch.utils.bottleneck
# srun -p gpu_p --gres=gpu:1 --qos=gpu_priority  --pty bash
# srun -p gpu_p --gres=gpu:1 --qos=gpu_priority --constraint=a100_40gb --pty bash
# python3 src/eval.py --train_set AML --predict resnet_next --criterion Focal --input_channels 3 --resizeW 64 --resizeH 64 --fold 1 --learning_rate 0.0004 --att_percent 0.1 --batch_size 1

def monitor_memory(pid, interval=1):
    """ Logs memory usage of a given process ID. """
    process = psutil.Process(pid)
    while True:
        mem_info = process.memory_info()
        print(f"[PID {pid}] RSS: {mem_info.rss / 1024**2:.2f} MB | VMS: {mem_info.vms / 1024**2:.2f} MB")
        time.sleep(interval)

def memory_hook(module, input, output):
    input_memory = sum([i.element_size() * i.nelement() for i in input]) / 1024  # Memory in KB
    output_memory = output.element_size() * output.nelement() / 1024  # Memory in KB
    print(f"Layer: {module.__class__.__name__}, "
          f"Input Memory: {input_memory:.4f} KB, "
          f"Output Memory: {output_memory:.4f} KB")
    
def register_hooks(model):
    hooks = []
    for layer in model.children():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            hook = layer.register_forward_hook(memory_hook)
            hooks.append(hook)
    return hooks

def main(args):
    num_classes = get_num_classes(args.train_set)

    # model_path = args.model_path
    # model_path_trim = model_path.split("/")[-2] 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # _, loader = Loader.load_datasets(args.train_set, args.fold, args.resizeW, args.resizeH, args.batch_size, args.criterion)

    for device in ["cuda"]: #,"cpu"
        print(f"Device: {device}")
        # print(f"Memory before loading model: {utils.get_memory_usage():.2f} MB")

        # # Start the memory monitor in a separate process
        # monitor_process = multiprocessing.Process(target=monitor_memory, args=(os.getpid(),), daemon=True)
        # monitor_process.start()

        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = False
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            utils.log_vram_usage("before inference")
        
        if args.predict == "resnet18":
            model = models.resnet18(weights=None)
            model.conv1 = torch.nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
            model = model.to(device)
        elif args.predict == "resnet_next":
            model = models.resnext50_32x4d(weights=None)
            model.conv1 = torch.nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
            model = model.to(device)
        elif args.predict == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.features[0][0] = torch.nn.Conv2d(args.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
            model = model.to(device)
        elif args.predict == "squeezenet":
            model = models.squeezenet1_1(weights=None)
            model.features[0] = torch.nn.Conv2d(args.input_channels, 64, kernel_size=3, stride=2, padding=1)
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
            model = model.to(device)
        elif args.predict == "shufflenet_v1":
            model = models.shufflenet_v2_x0_5(weights=None)
            model.conv1 = torch.nn.Conv2d(args.input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
            model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
            model = model.to(device)
        elif args.predict == "shufflenet_v2":
            model = models.shufflenet_v2_x0_5(weights=None)
            model.conv1 = torch.nn.Conv2d(args.input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
            model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes)
            model = model.to(device)
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

        # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        utils.log_vram_usage("after model loading")

        # print(f"Memory after loading model: {utils.get_memory_usage():.2f} MB")
        # return

        # print(f"Model loaded on {next(model.parameters()).device}")
        # utils.evaluate_model(loader, args.output + "_" + args.train_set, model, num_classes)
        # return
        # if "net" in args.predict:
        #     dummy_input = torch.randn(args.batch_size, args.input_channels, args.resizeH, args.resizeW).to(device) 
        # else:
        #     dummy_input = torch.randn(args.batch_size, args.resizeH, args.resizeW, args.channel_n_1).to(device) 
        # onnx_path = "wbcnca.onnx"
        # torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11)

        # print(f"Model saved as {onnx_path}")

        # return
        # utils.log_vram_usage("after loading model")

        with torch.inference_mode():
            adder = Adder()
            ####### warm up cuda
            if device.startswith("cuda"):
                for i in range(50):
                    if "net" in args.predict:
                        inputs = torch.randn(args.batch_size, args.input_channels, args.resizeH, args.resizeW).to(device) 
                    else:
                        inputs = torch.randn(args.batch_size, args.resizeH, args.resizeW, args.channel_n_1).to(device) 
                    model(inputs)

            for i in range(1000):
                if "net" in args.predict:
                    inputs = torch.randn(args.batch_size, args.input_channels, args.resizeH, args.resizeW).to(device) 
                else:
                    inputs = torch.randn(args.batch_size, args.resizeH, args.resizeW, args.channel_n_1).to(device) 
                    
                tm = time.time()
                
                model(inputs)
                
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                #     output = model(inputs)
                # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                # hooks = register_hooks(model)
                # for hook in hooks:
                #     hook.remove()
                elapsed = time.time() - tm
                adder(elapsed)

        avg_time = adder.average()/args.batch_size
        print("Average time: %.4f seconds" % avg_time)

    if device.startswith("cuda"):
        utils.log_vram_usage("after inference")
    
    return
    #### SINGLE IMAGE INFERENCE ########################################
    sample_index = 789
    if not os.path.exists("figs/{}".format(model_path_trim)):
        os.makedirs("figs/{}".format(model_path_trim))

    img, label = get_test_sample(args.resizeW, args.resizeH, index=sample_index, dataset=args.train_set)
    img_np = img.cpu().numpy()
    if args.input_channels == 1:
        utils.showimg(img_np, "figs/{}/input.png".format(model_path_trim), cmap="gray")
    else:
        utils.showimg(img_np, "figs/{}/input.png".format(model_path_trim))

    mask = None

    dict = model.state_dict()
    for k, v in dict.items():
        if "attention16" in k:
            v = F.sigmoid(v)
            mask16 = v.detach().cpu().numpy()  
            utils.showimg(mask16, "figs/{}/attention16.png".format(model_path_trim), cmap="gray")
        elif "attention" in k:
            v = F.sigmoid(v)
            mask = v.detach().cpu().numpy()  
            utils.showimg(mask, "figs/{}/attention.png".format(model_path_trim), cmap="gray")

    padding = torch.empty(args.resizeW, args.resizeH, args.channel_n_1-args.input_channels)
    img_padded = torch.cat([img, padding], dim=-1).to(device)

    with torch.no_grad():
        out = model(img_padded.unsqueeze(0))
    
    out = out.clone().detach().squeeze(0).cpu()
    out = torch.softmax(out, dim=-1)
    print("out: ", out)
    prediction = torch.argmax(out, dim=-1).item()
    max_prob = torch.max(out).item()
    prediction_2nd = torch.argsort(out, descending=True)[1].item()
    second_max_prob = torch.sort(out, descending=True)[0][1].item()
    
    print("prediction: ", prediction, "confidence: ", round(max_prob, 2))
    print("2nd prediction: ", prediction_2nd, "confidence: ", round(second_max_prob, 2))
    print("ground truth: ", label)

    max_memory_allocated = torch.cuda.max_memory_allocated() 
    max_memory_reserved = torch.cuda.max_memory_reserved()
    print(f"Max memory allocated during the run: {max_memory_allocated / 1024**2:.2f} MB ({max_memory_allocated / 1024**3:.2f} GB)")
    print(f"Max memory reserved during the run: {max_memory_reserved / 1024**2:.2f} MB ({max_memory_reserved / 1024**3:.2f} GB)")
    return

    # for input-dependent attention mask (mic)
    # if mask is None:
    #     x = feature.permute(0, 3, 1, 2)
    #     x_down = F.avg_pool2d(x, kernel_size=4)
    #     # x_down = F.max_pool2d(x, kernel_size=4)
    #     mask = model.conv1x1(x_down)
    #     # mask_sig_np = torch.sigmoid(mask).clone().detach().cpu().numpy()
    #     mask_np = mask.clone().squeeze(0).squeeze(0).detach().cpu().numpy()
    #     # showimg(mask_sig_np, "figs/{}/attention_sig.png".format(model_path_trim), cmap="gray") 
    #     showimg(mask_np, "figs/{}/attention.png".format(model_path_trim), cmap="gray")  

    #     attention = F.interpolate(mask, size=((args.resizeW, args.resizeH)), mode="bilinear", align_corners=False)  # (B, 1, 64, 64)
    #     attention_np = attention.clone().detach().squeeze(0).squeeze(0).cpu().numpy()
    #     showimg(attention_np, "figs/{}/attention_up.png".format(model_path_trim), cmap="gray")
    #     B, C, H, W = attention.shape
    #     attention = attention.contiguous().view(B, C, -1)
    #     attention = F.softmax(attention, dim=-1)
    #     mask = attention.view(B, C, H, W).squeeze(0).squeeze(0).detach().cpu().numpy()
    #     showimg(mask, "figs/{}/attention_up_softmax.png".format(model_path_trim), cmap="gray")  

    num_channels = feature.shape[-1]
    # sorted_indices = np.argsort(-input_weight_importance) 
    feature = feature.clone().detach().squeeze(0).cpu() # (64, 64, c)
    feature_sig_np = torch.sigmoid(feature).numpy()
    feature_np = feature.numpy()

    ######### for input-dependent attention mask (lastchannel)
    # if mask is None:
    #     mask = torch.sigmoid(feature[:, :, -1]).numpy()
    #     showimg(mask, "figs/{}/attention.png".format(model_path_trim), cmap="gray")

    fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    axes = axes.flatten()
    for i in range(num_channels):
        sorted_idx = i
        ax = axes[i]
        ax.imshow(feature_np[:, :, sorted_idx], cmap='coolwarm')
        ax.axis('off')
        ax.set_title(f'C {sorted_idx + 1}')
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/{}/channels_coolwarm.png".format(model_path_trim))
    plt.close()

    fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    axes = axes.flatten()
    for i in range(num_channels):
        sorted_idx = i
        ax = axes[i]
        ax.imshow(feature_sig_np[:, :, sorted_idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f'C {sorted_idx + 1}')
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/{}/channels_gray.png".format(model_path_trim))
    plt.close()

    feature_np = feature.numpy()
    feature_np_masked = feature_np.transpose(2, 0, 1) * mask

    # threshold = np.percentile(feature_np_masked, 90)
    # feature_np_masked = np.where(feature_np_masked >= threshold, feature_np_masked, 0).transpose(1, 2, 0)
    feature_np_masked = feature_np_masked.transpose(1, 2, 0)

    fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    axes = axes.flatten()
    for i in range(num_channels):
        # sorted_idx = sorted_indices[i] 
        sorted_idx = i
        activation = feature_np_masked[:, :, sorted_idx]
        activation_pil = Image.fromarray(activation.astype('float64'))
        activation = np.array(activation_pil)
        # activation_sig = torch.sigmoid(torch.tensor(activation)).numpy()
        ax = axes[i]
        ax.imshow(activation, cmap='coolwarm')
        ax.axis('off')
        ax.set_title(f'C {sorted_idx + 1}')
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/{}/channels_activated_coolwarm.png".format(model_path_trim))
    plt.close()

    fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    axes = axes.flatten()
    for i in range(num_channels):
        sorted_idx = i
        activation = feature_np_masked[:, :, sorted_idx]
        activation_pil = Image.fromarray(activation.astype('float64'))
        activation = np.array(activation_pil)
        # activation_sig = torch.sigmoid(torch.tensor(activation)).numpy()
        ax = axes[i]
        ax.imshow(activation, cmap='gray')
        ax.axis('off')
        ax.set_title(f'C {sorted_idx + 1}')
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/{}/channels_activated_gray.png".format(model_path_trim))
    plt.close()

    # if show_feature16:
    #     feature16_np = feature16.clone().detach().squeeze(0).cpu().numpy()
    #     feature16_np_masked = feature16_np * mask16
    #     threshold = np.percentile(feature16_np_masked, 90)
    #     feature16_np_masked = np.where(feature16_np_masked >= threshold, feature16_np_masked, 0).transpose(1, 2, 0)

    #     fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    #     axes = axes.flatten()
    #     for i in range(num_channels):
    #         sorted_idx = sorted_indices[i] 

    #         activation = feature16_np_masked[:, :, sorted_idx]
    #         activation_pil = Image.fromarray(activation.astype('float64'))
    #         resize_transform = transforms.Resize((args.resizeW, args.resizeH))  
    #         activation = resize_transform(activation_pil)
    #         activation = np.array(activation)
    #         activation_norm = (activation - np.min(activation)) / (np.max(activation) - np.min(activation) + 1e-6)

    #         cmap = plt.get_cmap('hot')
    #         norm = Normalize(vmin=0, vmax=1)
    #         scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)
    #         heatmap_rgb = scalar_mappable.to_rgba(activation_norm, bytes=True)  # Shape: (height, width, 4)
    #         activation = heatmap_rgb[..., :3] / 255.0
    #         overlay = 0.3 * img_norm + 0.7 * activation
            
    #         ax = axes[i]
    #         ax.imshow(overlay) 
    #         importance_value = importance[sorted_idx]
    #         importance_color = (importance_value, 0, 1 - importance_value, 1)  # Red (high) to Blue (low)
    #         rect = patches.Rectangle(
    #             (-2, -2),  
    #             img_norm.shape[1], img_norm.shape[0],  
    #             linewidth=20,  
    #             edgecolor=importance_color,
    #             facecolor='none'
    #         )
    #         ax.add_patch(rect)
    #         ax.axis('off')
    #         ax.set_title(f'C {sorted_idx + 1}')
    #     for j in range(num_channels, len(axes)):
    #         axes[j].axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig("figs/{}/channel_activations16.png".format(model_path_trim))
    #     plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--predict', type=str)
    parser.add_argument('--criterion', type=str, default="Focal")
    parser.add_argument('--input_channels', type=int)
    parser.add_argument('--resizeW', type=int, default=64)
    parser.add_argument('--resizeH', type=int, default=64)
    parser.add_argument('--att_percent', type=float, default=0.1)
    parser.add_argument('--steps_1', type=int, default=64)
    parser.add_argument('--channel_n_1', type=int, default=128)
    parser.add_argument('--hidden_size_1', type=int, default=128)
    parser.add_argument('--steps_2', type=int, default=32)
    parser.add_argument('--channel_n_2', type=int, default=128)
    parser.add_argument('--hidden_size_2', type=int, default=128)
    parser.add_argument('--hidden_size_fcn', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=4e-4) 
    parser.add_argument('--n_epochs', type=int, default=32)
    parser.add_argument('--fire_rate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    main(args)