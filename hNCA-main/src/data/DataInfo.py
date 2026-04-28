AML_data_path = ""
PBC_data_path = ""
MLL_data_path = ""
Malaria_data_path = ""
SIPAKMED_data_path = ""
Urine_data_path = ""

num_classes_AML = 13
num_classes_PBC = 13
num_classes_MLL = 13
num_classes_Malaria = 2
num_classes_SIPAKMED = 5
num_classes_Urine = 8

def get_num_classes(dataset):
    if dataset == "AML":
        return num_classes_AML
    elif dataset == "PBC":
        return num_classes_PBC
    elif dataset == "MLL":
        return num_classes_MLL
    elif dataset == "Malaria":
        return num_classes_Malaria
    elif dataset == "SIPAKMED":
        return num_classes_SIPAKMED
    elif dataset == "Urine":
        return num_classes_Urine
    else:
        raise ValueError("Unknown dataset: " + dataset)