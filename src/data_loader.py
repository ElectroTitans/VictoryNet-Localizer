from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np

def get_data():
    print("[LocalNet / DataLoader] Loading Training Data... ")
    path = 'Data/'  # use your path
    allFiles = glob.glob(path + "/*")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_, ignore_index=True)

    frame_train, frame_test = train_test_split(frame, test_size=0.3)

    print("[LocalNet / DataLoader] Loaded Training and Testing Data: " + str(len(frame_train)) + "/" + str(len(frame_test)))
    
    return frame_train, frame_test


def format_data(data):
    x2 = data[["Heading"]]
    y = data[["X-Coord", "Y-Coord"]]

    data.drop(["X-Coord", "Y-Coord", "Heading"], axis=1, inplace=True)
    x1 = np.expand_dims(data, axis=2)

    return x1, x2, y 
