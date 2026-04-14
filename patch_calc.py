import math

def out_dim(in_dim, kernel, stride=1, pad=0):
    return math.floor((in_dim + 2*pad - kernel) / stride) + 1

x = 32

# conv + pool repeated 4 times
for i in range(4):
    x = out_dim(x, kernel=3, stride=1, pad=1)   # conv
    print(f"after conv {i+1}: {x}")
    x = out_dim(x, kernel=2, stride=2, pad=0)   # pool
    print(f"after pool {i+1}: {x}")