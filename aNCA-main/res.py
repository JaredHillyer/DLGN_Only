import pickle
import numpy as np

with open('res.pkl', 'rb') as f:
    data = pickle.load(f)

# keys_to_delete = []
# exprs = ["_1_64_64_AML"]
# for key in list(data.keys()):
#     # if key.startswith(expr):
#     if any(expr in key for expr in exprs):
#         keys_to_delete.append(key)

# for key in keys_to_delete:
#     data.pop(key, None)

data = dict(sorted(data.items()))

with open('res.pkl', 'wb') as f:
    pickle.dump(data, f)


with open('res.pkl', 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    print(key)

experiments = []

def get_average(data, experiment, val=""):
    sum_acc = 0
    sum_b_acc = 0
    sum_f1 = 0
    num = 0

    for key in data.keys():
        val = ""
        if key.split('_')[-1] == "val":
            continue
        if key.split('_')[-1] == "val" or key.split('_')[-1] == "test":
            val = key.split('_')[-1]
        if key.split('+')[0] == experiment:# and  key.split('_')[-1] == val:
            fold = key.split('+')[-1].split('_')[0]
            # print(f"fold {fold} {val}: acc: {data[key]['accuracy']*100:.2f}%, bacc: {data[key]['b_accuracy']*100:.2f}%, f1: {data[key]['f1']*100:.2f}%")
            sum_acc += data[key]['accuracy']
            sum_b_acc += data[key]['b_accuracy']
            sum_f1 += data[key]['f1']
            num += 1
    
    std_acc = np.std([data[key]['accuracy'] for key in data.keys() if key.split('+')[0] == experiment])# and  key.split('_')[-1] == val])
    std_b_acc = np.std([data[key]['b_accuracy'] for key in data.keys() if key.split('+')[0] == experiment])# and  key.split('_')[-1] == val])
    std_f1 = np.std([data[key]['f1'] for key in data.keys() if key.split('+')[0] == experiment])# and  key.split('_')[-1] == val])
    return sum_acc/num, sum_b_acc/num, sum_f1/num, std_acc, std_b_acc, std_f1

#find all experimets that contains "OCT", and store them in a list, remove duplicates
experiments = [key.split('+')[0] for key in data.keys() if "OCT" in key]
experiments = list(set(experiments))
experiments.sort()

for experiment in experiments:
    print("******************************************************************************************")
    print(f"Experiment: {experiment}\n")
    acc, bacc, f1, std_acc, std_bacc, std_f1 = get_average(data, experiment)
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Average acc: {acc*100:.1f} ± {std_acc*100:.1f}, f1: {f1*100:.1f} ± {std_f1*100:.1f}\n")
    print(f"!!!!!!!!!!!!!!!!!Average acc: {acc*100:.1f} ± {std_acc*100:.1f}, bacc: {bacc*100:.1f} ± {std_bacc*100:.1f}, f1: {f1*100:.1f} ± {std_f1*100:.1f}\n")