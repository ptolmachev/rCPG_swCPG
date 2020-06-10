import json
from utils.gen_utils import get_project_root, create_dir_if_not_exist

data_folder = f"{get_project_root()}/data"

default_neural_params = {
    'C' : 20,
    'g_NaP' : 0.0,
    'g_K' : 5.0,
    'g_ad' : 10.0,
    'g_l' : 2.8,
    'g_synE' : 10,
    'g_synI' : 60,
    'E_Na' : 50,
    'E_K' : -85,
    'E_ad' : -85,
    'E_l' : -60,
    'E_synE' : 0,
    'E_synI' : -75,
    'V_half' : -30,
    'slope' : 4,
    'tau_ad' : 2000,
    'K_ad' : 0.9,
    'tau_NaP_max' : 6000}
create_dir_if_not_exist(f'{data_folder}/params')
json.dump(default_neural_params, open(f'{data_folder}/params/default_neural_params.json', 'w+'))