import yaml
from pprint import pprint
import os
import json
def get_model_config():
    print("[LocalNet / ConfigLoader] Loading Model Config from 'settings.yaml'...")
    with open("settings.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    model_name = "LocalNet " + str(cfg["model_version"]) + "_" + str(cfg["model_edit"]) + "_" + str( cfg["model_run"])
    print("[LocalNet / ConfigLoader] Loaded Settings for model: " + model_name)
    pprint(cfg)
    cfg["model_name"] = model_name
    cfg["filepath_root"] = "./Graph/"+model_name+"/"
    cfg["filepath_weights"] = "./Graph/"+model_name+"/"+model_name+"_trained.hdf5"

    return model_name, cfg

def get_env_config():
    print("[LocalNet / ConfigLoader] Loading Env Config from 'Data/_settings.json'...")
    with open('Data/_settings.json') as f:
        env_settings = json.load(f)
    print("[LocalNet / ConfigLoader] Loaded Settings for enviorment training set: " )
    pprint(env_settings)
    return env_settings

def backup():
    print("[LocalNet / ConfigLoader] Backing up settings")
    if not os.path.exists( "./Graph/"+model_name+"/"):
        os.makedirs( "./Graph/"+model_name+"/")
    print("[LocalNet / ConfigLoader] Backed up settings")