import json
import numpy as np

# ----------------------------------------
# script to create json config file
# --------------

data = {}

data_agnostic_configs = {
    "demo": "True",
    "dataset": "mnist",
    "basis": "csdip",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "number_iterations": 300,
    "number_restarts": 5,
    "measure_type": "phase_retrieval"   # ["phase_retrieval", "lin_comp_sense" ]
}

xray = {
    "img_size": 256,
    "num_channels": 1,
    "num_measurements": 1000,
    "z_dim": 32
}

mnist = {
    "img_size": 28,
    "num_channels": 1,
    "z_dim": 128
}

celeba = {
    "img_size": 64,
    "num_channels": 1,
    "z_dim": 100
}


# define list of measurement sizes
# mn_ratios = np.asarray([0.95])
mn_ratios = np.concatenate([np.linspace(0.05, 0.17, 4), np.linspace(0.18, 0.3, 7), np.linspace(0.35,0.9, 3)])
# mn_ratios = np.linspace(0.05, 0.9, 4)


data_length = 28*28
measurement_array = mn_ratios*data_length
measurement_list = np.ndarray.tolist(measurement_array.astype(int))

mnist["num_measurements"] = measurement_list

data_length = 64*64
measurement_array = mn_ratios*data_length
measurement_list = np.ndarray.tolist(measurement_array.astype(int))
celeba["num_measurements"] = measurement_list


# put all parsing info into 'data' object

data["data_agnostic_configs"] = data_agnostic_configs
data['xray'] = xray
data['mnist'] = mnist
data['celeba'] = celeba

# save config file

with open('configs.json', 'w') as outfile:
    json.dump(data, outfile)