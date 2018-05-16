from src.config_loader import get_model_config, get_env_config, backup
from src.data_loader import get_data, format_data
from src.model import make_model
from keras.callbacks import ModelCheckpoint
import keras
model_name, model_cfg = get_model_config()
env_cfg = get_env_config()

frame_train, frame_test = get_data()

x1_train, x2_train, y_train = format_data(frame_train)
x1_test,  x2_test,  y_test  = format_data(frame_test)

model = make_model(model_cfg, env_cfg)

checkpoint = ModelCheckpoint(model_cfg['filepath_root'],
                            monitor='loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')

embedding_layer_names = [ 'lidar_conv1']


tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph/'+model_name, 
    histogram_freq=5,
    write_grads=True,
    write_graph=True,
    batch_size=model_cfg["batch_size"])

model.fit([x1_train, x2_train], y_train,
          batch_size=model_cfg["batch_size"],
          epochs=model_cfg["epoch"],
          verbose=1,
          shuffle=True,
          validation_data=([x1_test, x2_test], y_test),
          callbacks=[checkpoint,tbCallBack])


score = model.evaluate([x1_test, x2_test], y_test, verbose=1)