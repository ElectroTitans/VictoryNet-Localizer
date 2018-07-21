import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input, Add, Concatenate, SimpleRNN, GRU, BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

def make_model(model_cfg, env_cfg):
    if model_cfg["load"] is True:
        print("[LocalNet / Model] Loading Previous Model from: " + model_cfg['filepath_trained'])
        model = load_model(filepath)
        print(model.summary())

    else:
        print("[LocalNet / Model] Creating Model")
        lidar_input       = Input(
                                shape=(env_cfg['lineNum'],1), 
                                name='lidar_input'
                            )
        
        lidar_conv1       = Conv1D(
                                model_cfg["conv1_filter"], 
                                kernel_size=model_cfg["conv1_kernal"], 
                                strides=(1),
                                activation='relu', 
                                name='lidar_conv1'
                            )(lidar_input)
        lidar_batch1      = BatchNormalization(name='lidar_batch1')(lidar_conv1)
        lidar_pooling1    = AveragePooling1D(
                                pool_size=(2), 
                                strides=(2),  
                                name='lidar_pooling1'
                            )(lidar_batch1)

        lidar_conv2       = Conv1D(
                                model_cfg["conv2_filter"],
                                kernel_size=model_cfg["conv2_kernal"], 
                                activation='relu',  
                                name='lidar_conv2'
                            )(lidar_pooling1)
        lidar_batch2      = BatchNormalization(name='lidar_batch2')(lidar_conv2)
        lidar_pooling2    = AveragePooling1D(
                                pool_size=(2),  
                                name='lidar_pooling2'
                            )(lidar_batch2)

        lidar_conv3       = Conv1D(
                                model_cfg["conv3_filter"],
                                kernel_size=model_cfg["conv3_kernal"], 
                                activation='relu',  
                                name='lidar_conv3'
                            )(lidar_pooling2)

        lidar_batch3      = BatchNormalization(name='lidar_batch3')(lidar_conv3)

        lidar_pooling3    = AveragePooling1D(
                                pool_size=(2),  
                                name='lidar_pooling3'
                            )(lidar_batch3)


        lidar_flatten     = Flatten( 
                                name='lidar_flatten'
                            )(lidar_pooling3)

        imu_input         = Input(
                                shape=(1,) ,
                                name='imu_input'
                            )

        combined_layer    = Concatenate(
                                name='combined_layer'
                            )([lidar_flatten, imu_input])

        final_dense       = Dense(
                                model_cfg["fully_connected"], 
                                activation='relu', 
                                name='final_dense'
                            )(combined_layer)

        


        coord_out         = Dense(2, 
                                name='coord_out'
                            )(final_dense)

        model = Model(inputs=[lidar_input, imu_input], outputs=coord_out)

        print(model.summary())
        print("[LocalNet / Model] Compliling Model")

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(
                lr=model_cfg['learning_rate']
            )
        )

    return model
