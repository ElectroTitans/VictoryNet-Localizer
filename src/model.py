import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input, Add, Concatenate, SimpleRNN
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

        lidar_pooling1    = AveragePooling1D(
                                pool_size=(2), 
                                strides=(2),  
                                name='lidar_pooling1'
                            )(lidar_conv1)

        lidar_conv2       = Conv1D(
                                model_cfg["conv2_filter"],
                                kernel_size=model_cfg["conv2_kernal"], 
                                activation='relu',  
                                name='lidar_conv2'
                            )(lidar_pooling1)

        lidar_pooling2    = AveragePooling1D(
                                pool_size=(2),  
                                name='lidar_pooling2'
                            )(lidar_conv2)

        lidar_flatten     = Flatten( 
                                name='lidar_flatten'
                            )(lidar_pooling2)

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

        rnn               = SimpleRNN(
                                32, 
                                activation='tanh', 
                                use_bias=True, 
                                kernel_initializer='glorot_uniform', 
                                recurrent_initializer='orthogonal', 
                                bias_initializer='zeros', 
                                kernel_regularizer=None, 
                                recurrent_regularizer=None, 
                                bias_regularizer=None, 
                                activity_regularizer=None, 
                                kernel_constraint=None, 
                                recurrent_constraint=None, 
                                bias_constraint=None, 
                                dropout=0.0, 
                                recurrent_dropout=0.0, 
                                return_sequences=False, 
                                return_state=False, 
                                go_backwards=False, 
                                stateful=False, 
                                unroll=False
                            )(final_dense)


        coord_out         = Dense(2, 
                                name='coord_out'
                            )(rnn)

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