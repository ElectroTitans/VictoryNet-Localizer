from google.cloud import datastore
# Imports the Google Cloud client library
from google.cloud import storage

import socket

import keras
import datetime
import zipfile
import os

datastore_client = datastore.Client('victory-net')

def init_model(model_name):
    print('[LocalNet / GCP] Init GCP Datastore')
    kind = 'Model'
    # The name/ID for the new entity
    name = model_name
    # The Cloud Datastore key for the new entity
    task_key = datastore_client.key(kind, name)
    print('[LocalNet / GCP] Datastore Key Made: ' + str(task_key))
    return datastore.Entity(key=task_key)

def set_cfgs(task, model_cfg, env_cfg):
    print('[LocalNet / GCP] Setting Cfgs')
    print(socket.gethostname())
    task['info_machine'] = socket.gethostname()
    task['info_date']  = datetime.datetime.now()
    task['info_name']  = model_cfg['model_name']
    task['info_desc']  = model_cfg['model_desc']
    task['info_status'] = "Loading Training Data"
    task['cfg_model_version']   = model_cfg['model_version']
    task['cfg_model_edit']      = model_cfg['model_edit']
    task['cfg_model_run']       = model_cfg['model_run']
    task['cfg_learning_rate']   = model_cfg['learning_rate']
    task['cfg_batch_size']      = model_cfg['batch_size']
    task['cfg_epoch']           = model_cfg['epoch']
    task['cfg_conv1_filter']    = model_cfg['conv1_filter']
    task['cfg_conv2_filter']    = model_cfg['conv2_filter']
    task['cfg_conv1_kernal']     = model_cfg['conv1_kernal']
    task['cfg_conv2_kernal']     = model_cfg['conv2_kernal']
    task['cfg_fully_connected'] = model_cfg['fully_connected']


    task['env_lineNum']      = env_cfg['lineNum']
    task['env_noise']        = env_cfg['noise']
    task['env_dropout']      = env_cfg['dropout']
    task['env_maxRange']     = env_cfg['maxRange']
    task['env_spinRate']     = env_cfg['spinRate']
    task['env_instantMode']  = env_cfg['instantMode']

    # Saves the entity
    datastore_client.put(task)

    print('[LocalNet / GCP] Set Configs')

    
def set_dataset(task ,train_len, val_len, name="Dataset"):
    print('[LocalNet / GCP] Setting Dataset Info')
    task['info_dataset'] = name
    task['info_training_length'] = train_len
    task['info_testing_length'] = val_len
    datastore_client.put(task)
    print('[LocalNet / GCP] Set Dataset Info')

def validate_dataset(dataset_name):
    print("[LocalNet / GCP] Checking for dataset: " + dataset_name)
   
    if not os.path.exists( "./Data/" + dataset_name):
        print("Making Dataset Folder.")
        os.makedirs( "./Data/"+ dataset_name)
        download_dataset(dataset_name)
       

def download_dataset(name): 
    print("[LocalNet / GCP] Downloading Dataset: " + name)
    storage_client = storage.Client("victorynet")
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket("victorynet-trainingdata")
    # Create a blob object from the filepath
    blob = bucket.blob(name+".zip")
    # Download the file to a destination
    blob.download_to_filename(name+".zip")
    print("Downloaded! Unzipping")
    zip_ref = zipfile.ZipFile(name+".zip", 'r')
    zip_ref.extractall("Data/"+name)
    zip_ref.close()

def set_status(task, status):
    print('[LocalNet / GCP] Setting Status: ' + status)
    task['info_status'] = status
    datastore_client.put(task)
    print('[LocalNet / GCP] Set Status!')


class GCPDatastoreCheckpoint(keras.callbacks.Callback):
    def __init__(self, task, cfg):
        self.task = task
        self.cfg = cfg
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.task['info_epoch'] = epoch + 1
        self.task['info_status'] = "Training"
        self.task['info_progress'] = (epoch + 1) / self.cfg['epoch']
        self.task['info_loss'] = logs.get('loss')
        self.task['info_val_loss'] = logs.get('val_loss')
        self.task['info_losses'] = self.losses
        datastore_client.put(self.task)
        print('Updated to GCP Datastore{}'.format(self.task.key))
        
        upload_blob("victorynet-models", self.cfg["filepath_weights"], self.cfg["model_name"] + "_trained.hdf5")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client('victory-net')
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))