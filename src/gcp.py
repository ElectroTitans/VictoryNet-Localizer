def init_model(model_settings, env_settings):

def submit_dataset(train_len, val_len):

class GCPDatastoreCheckpoint(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        task['info_epoch'] = epoch + 1
        task['info_status'] = "Training"
        task['info_progress'] = (epoch + 1) / cfg['epoch']
        task['info_loss'] = logs.get('loss')
        task['info_val_loss'] = logs.get('val_loss')
        task['info_losses'] = self.losses
        datastore_client.put(task)
        print('Updated to GCP Datastore{}'.format(task.key))
