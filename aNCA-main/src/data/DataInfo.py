#AML_data_path = "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Matek_cropped/"
#PBC_data_path = "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped/"
#MLL_data_path = "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/MLL_20221220/"
AML_data_path = "/lustre/groups/aih/michael.deutges/Datasets/Cytomorphology_Matek/AML-Cytomorphology_LMU/"
PBC_data_path = "/lustre/groups/aih/michael.deutges/Datasets/Acevedo_Barcelona_2020/PBC_divided/"
MLL_data_path = "/lustre/groups/aih/michael.deutges/Datasets/mll_1/mll_folderwise/"
Malaria_data_path = "/lustre/groups/aih/michael.deutges/Datasets/Malaria/cell_images"
SIPAKMED_data_path = "/lustre/groups/aih/michael.deutges/Datasets/SIPaKMeD/original_files"
Urine_data_path = "/lustre/groups/aih/michael.deutges/Datasets/Urine_Sediments/urine_data"
CIFAR10_data_path = "/lustre/groups/aih/chen.yang/CIFAR10"
CRC_data_path = "/lustre/groups/aih/chen.yang/CRC"
PatchCamelyon_data_path = "/lustre/groups/aih/chen.yang/PatchCamelyon"
OCT_data_path = "/lustre/groups/aih/chen.yang/OCT/OCT_Classification"

num_classes_AML = 13
num_classes_PBC = 13
num_classes_MLL = 13
num_classes_Malaria = 2
num_classes_SIPAKMED = 5
num_classes_Urine = 8
num_classes_CIFAR10 = 10
num_classes_CRC = 9
num_classes_CRC_nonorm = 9
num_classes_PatchCamelyon = 2
num_classes_OCT = 12

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
    elif dataset == "CIFAR10":
        return num_classes_CIFAR10
    elif dataset == "CRC":
        return num_classes_CRC
    elif dataset == "CRC_nonorm":
        return num_classes_CRC_nonorm
    elif dataset == "PatchCamelyon":
        return num_classes_PatchCamelyon
    elif dataset == "OCT":
        return num_classes_OCT
    else:
        raise ValueError("Unknown dataset: " + dataset)