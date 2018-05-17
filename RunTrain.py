from src.config_loader import get_model_config, get_env_config, backup
from src.data_loader import get_data, format_data
from src.model import make_model
from src import gcp

from keras.callbacks import ModelCheckpoint
import keras
import argparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config File to use")
parser.add_argument("--gpu", help="GPU Usage 0.1-1.0", type=float)
args = parser.parse_args()
print(args)
cfg="settings.yaml"
if(args.config):
    print("Selecting Config: " + args.config)
    cfg = args.config
if(args.gpu):
    print("Running GPU%:  "+args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu
    set_session(tf.Session(config=config))

model_name, model_cfg = get_model_config(cfg)

gcp.validate_dataset(model_cfg['dataset'])

env_cfg = get_env_config(model_cfg['dataset'])



key = gcp.init_model(model_name)
gcp.set_cfgs(key, model_cfg, env_cfg)

frame_train, frame_test = get_data(model_cfg['dataset'])

gcp.set_dataset(key, len(frame_train), len(frame_test), name=model_cfg['dataset'])

x1_train, x2_train, y_train = format_data(frame_train)
x1_test,  x2_test,  y_test  = format_data(frame_test)

gcp.set_status(key, "Compiling Model")

model = make_model(model_cfg, env_cfg)

checkpoint = ModelCheckpoint(model_cfg['filepath_weights'],
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

embedding_layer_names = [ 'lidar_conv1']

datastoreCB = gcp.GCPDatastoreCheckpoint(key, model_cfg)

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph/'+model_name, 
    histogram_freq=5,
    write_grads=True,
    write_graph=True,
    batch_size=model_cfg["batch_size"])

gcp.set_status(key, "Training First Epoch")

model.fit([x1_train, x2_train], y_train,
          batch_size=model_cfg["batch_size"],
          epochs=model_cfg["epoch"],
          verbose=1,
          shuffle=True,
          validation_data=([x1_test, x2_test], y_test),
          callbacks=[checkpoint,datastoreCB, tbCallBack])

gcp.set_status(key, "Evalulating")
score = model.evaluate([x1_test, x2_test], y_test, verbose=1)
gcp.set_status(key, "Complete")