import yaml
def get_model_config():
    print("[LocalNet] Loading Model Config from 'settings.yaml'...")
    with open("settings.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    model_name = "LocalNet " + str(cfg["model_version"]) + "_" + str(cfg["model_edit"]) + "_" + str( cfg["model_run"])
    print("[LocalNet] Loaded Settings for model: " + mode_name)
    return model_name, cfg

