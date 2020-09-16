import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# plot curve with max accuracies
def plot(data, n_seeds):
    x = np.array(data[:,0])
    y1_max = np.array(data[:,1])
    y2_max = np.array(data[:,2])

    plt.plot(x, y1_max, label='Drugs and Distractors')
    plt.plot(x, y2_max, label='Only drus')
    plt.xlabel('Dataset size (per drug)')
    plt.ylabel('maximum accuracy')
    plt.legend()
    plt.title("PADs CNN - Ensembled test (%s seeds by dataset size)" % (n_seeds))
    plt.yticks(np.arange(60, max(y2_max)+5, 4.0))
    plt.grid(True)
    plt.show()

# CSV reader
def load_csv(fname):
    df = pd.read_csv(fname, sep=',',header=0)
    data = df.values        
    return data, list(df)
    