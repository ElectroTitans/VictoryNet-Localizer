from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, Input, Add, Concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.utils import np_utils

def make_model(model_settings, env_settings):
