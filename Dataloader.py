import os
try:
    print('SlURM_JOB_ID',os.environ["SLURM_JOB_ID"])
except:
    print("no slurm id")
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tables
import gc
import argparse
import scipy

tf.keras.backend.set_epsilon(0.0000001)

def get_data_GE(datapath, setname, fold):
    ytest = pd.read_csv(datapath + "y" + str(setname) + "_" + gt_name + "_"+str(fold)+".csv")
    h5file = tables.open_file(datapath + "GE_CPM_nonorm_overlap_transposed.h5", "r")
    ybatch = ytest["labels"]
    xbatchid = np.array(ytest["expr_row"].values, dtype=np.int64)
    xbatch = h5file.root.data[xbatchid, :]
    ybatch = np.reshape(np.array(ybatch), (-1, 1))
    h5file.close()
    return (xbatch, ybatch)


def get_data_ME(datapath, setname, fold):
    ytest = pd.read_csv(datapath + "y" + str(setname) + "_" + gt_name + "_"+str(fold)+".csv")
    h5file = tables.open_file(datapath + "ME_transposed.h5", "r")
    ybatch = ytest["labels"]
    xbatchid = np.array(ytest["meth_row"].values, dtype=np.int64)
    xbatch = h5file.root.data[xbatchid, :]
    ybatch = np.reshape(np.array(ybatch), (-1, 1))
    h5file.close()
    return (xbatch, ybatch)