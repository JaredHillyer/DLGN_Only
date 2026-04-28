import onnxruntime as ort
import numpy as np
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())  # Get current process
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

# Load ONNX model
onnx_model_path = "nca.onnx"
session = ort.InferenceSession(onnx_model_path)
input = np.random.randn(1, 64, 64, 128).astype(np.float32)
# input = np.random.randn(1, 3, 64, 64).astype(np.float32)

# sample_index = 789
# img, label = get_test_sample(args.resizeW, args.resizeH, index=sample_index, dataset=args.train_set)
# img_np = img.cpu().numpy()
# if args.input_channels == 1:
#     utils.showimg(img_np, "figs/{}/input.png".format(model_path_trim), cmap="gray")
# else:
#     utils.showimg(img_np, "figs/{}/input.png".format(model_path_trim))

# padding = torch.empty(args.resizeW, args.resizeH, args.channel_n_1-args.input_channels)
# img_padded = torch.cat([img, padding], dim=-1).to(device)

# Check memory before inference
mem_before = get_memory_usage()
print(f"Memory usage before inference: {mem_before:.2f} MB")

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
out = session.run([output_name], {input_name: input})[0]
out = out.squeeze(0)
out = np.exp(out) / np.sum(np.exp(out))
print("out: ", out)
prediction = np.argmax(out)
max_prob = np.max(out).item()
print("prediction: ", prediction, ". confidence: ", round(max_prob, 2))

# Check memory after inference
mem_after = get_memory_usage()
print(f"Memory usage after inference: {mem_after:.2f} MB")

# Calculate memory used by inference
mem_used = mem_after - mem_before
print(f"Memory used during inference: {mem_used:.2f} MB")
