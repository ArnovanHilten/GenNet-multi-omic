import os
import argparse
try:
    print('SlURM_JOB_ID',os.environ["SLURM_JOB_ID"])
except:
    print("no slurm id")
import matplotlib
import scipy
import tables
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import LocallyDirectedConnected_tf2 as LocallyDirectedConnected
import sklearn.metrics as skm
from models.ClassificationModels import *
tf.keras.backend.set_epsilon(0.0000001)


def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(y_true, 0.0001, 1)
    y_pred = K.backend.clip(y_pred, 0.0001, 1)

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_possitive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)


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

class dataset_generator:
    
    def __init__(self, datapath, batchsize, setname, fold):
        self.batchsize = batchsize
        self.datapath = datapath
        self.setname = setname
        self.fold = fold
    
    def generate_batch(self):
        yoverview = pd.read_csv(self.datapath + "y" + str(self.setname) + "_" + gt_name + "_"+str(self.fold)+".csv")
        ylabels = np.reshape(np.array(yoverview["labels"]), (-1, 1))
        
        patient_ids = np.arange(yoverview.shape[0])
        np.random.shuffle(patient_ids)
        
        h5file_GE = tables.open_file(self.datapath + "GE_CPM_overlap_transposed.h5", "r")
        h5file_ME = tables.open_file(self.datapath + "ME_transposed.h5", "r")

        for xbatchid in np.array_split(patient_ids, indices_or_sections= batchsize):
            
            xbatchid_ME = np.array(yoverview.iloc[xbatchid]["meth_row"].values, dtype=np.int64, copy=True)
            xbatchid_GE = np.array(yoverview.iloc[xbatchid]["expr_row"].values, dtype=np.int64, copy=True)
            
            xbatch_GE = np.array(h5file_GE.root.data[xbatchid_GE, :])
            xbatch_ME = np.array(h5file_ME.root.data[xbatchid_ME, :])
            ybatch = ylabels[xbatchid]
            yield [xbatch_GE, xbatch_ME], ybatch 
            
        h5file_GE.close()
        h5file_ME.close()
        print("file closed")
    


   
def evaluate_performance(y, p):
    print("\n")
    print("Confusion matrix")
    confusion_matrix = skm.confusion_matrix(y, p.round())
    print(confusion_matrix)

    fpr, tpr, thresholds = skm.roc_curve(y, p)
    roc_auc = skm.auc(fpr, tpr)
    print("\n")
    print("Area under the Curve (AUC) = ", roc_auc)

    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    print('Specificity = ', specificity)

    sensitivity = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    print('Sensitivity = ', sensitivity)
    print("F_1 score = " + str(skm.f1_score(y, p.round())))
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(1 - specificity, sensitivity, color='b', marker='o')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    return roc_auc, confusion_matrix


def main(jobid, lr_opt, batch_size, l1_value, modeltype, pheno_name, fold):
    print(tf.__version__)
    global gt_name, datapath, weight_possitive_class, weight_negative_class
    print(jobid)
    jobid = int(jobid)
    lr_opt = float(lr_opt)
    batch_size = int(batch_size)
    l1_value = float(l1_value)
    fold = int(fold)
    datapath = "/trinity/home/avanhilten/repositories/multi-omics/bios/processed_data/"
    epochs = 2000
    dropout = 0
    extraname = ""
    augment = False
    gt_name = pheno_name
    weight_negative_class = 1
    second_run = False
    wpc = 1

    print(modeltype)
    
    if lr_opt == 0:
        optimizer = tf.keras.optimizers.Adadelta()
        print("adadelta")
    else:
        optimizer = tf.keras.optimizers.Adam(lr=lr_opt)
        print("Adam", lr_opt)

    if wpc == 1:
        ytrain = pd.read_csv(datapath + "ytrain" + "_" + gt_name + "_"+str(fold)+".csv")
        print('number with 1 in train= ', sum(ytrain["labels"]))
        print('number with 0 in train= ', sum(ytrain["labels"] == 0))
        weight_possitive_class = 1 / (ytrain["labels"].sum() / ytrain.shape[0])
        print('weight_possitive_class', weight_possitive_class)
    else:
        weight_possitive_class = wpc

    if modeltype == "single_input_Lasso_ge":
        xtrain, ytrain = get_data_GE(datapath, 'train', fold)
        xval, yval = get_data_GE(datapath, "val", fold)
    if (modeltype == "single_input_Lasso_me" )|(modeltype == "Gene_me"):
        xtrain, ytrain = get_data_ME(datapath, 'train', fold)
        xval, yval = get_data_ME(datapath, "val", fold)
    
    

    folder = ("Results_" + str(gt_name) + "__" + str(jobid) + "_fold_" + str(fold))

    inputsize = xtrain.shape[1]


    rfrun_path = "/trinity/home/avanhilten/repositories/multi-omics/bios/results/MEGE_" + folder + "/"
    if not os.path.exists(rfrun_path):
        print("Runpath did not exist but is made now")
        os.mkdir(rfrun_path)

    try:    
        with open(rfrun_path + '/Slurm_'+str(os.environ["SLURM_JOB_ID"])+'_.txt', 'w') as f:
            f.write('Slurm_ = ' + str(os.environ["SLURM_JOB_ID"]))
    except:
        print("no slurm job id")
        
    print("jobid =  " + str(jobid))
    print("fold =  " + str(fold))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))

                                           
 

    if modeltype == "single_input_Lasso_ge":
        model = single_input_Lasso_ge(inputsize_GE=int(inputsize), l1_value=l1_value)   
    if modeltype == "single_input_Lasso_me":
        model = single_input_Lasso_me(inputsize_ME=int(inputsize), l1_value=l1_value)    
    if modeltype == "single_input_Gene_me":
        model = single_input_Gene_me(inputsize_ME=int(inputsize), l1_value=l1_value) 
        
    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    print(model.summary())

    with open(rfrun_path + '/experiment_stats_results_.txt', 'a') as f:
        f.write('gtname = ' + str(gt_name))
        f.write('\n')
        f.write('\n')
        f.write('\n jobid = ' + str(jobid))
        f.write('\n model = ' + str(modeltype))
        f.write('\n batchsize = ' + str(batch_size))
        f.write('\n dropout = ' + str(dropout))
        f.write('\n extra = ' + str(extraname))
        f.write('\n augment = ' + str(augment))

    with open(rfrun_path + '/experiment_summary_model.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    csv_logger = K.callbacks.CSVLogger(rfrun_path + 'log.csv', append=True, separator=';')
    earlystop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=1, mode='auto')
    saveBestModel = K.callbacks.ModelCheckpoint(rfrun_path + "bestweight_job.h5", monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='auto')
    reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=5, min_lr=0.001)
    
    
    
    
    # %%
    if os.path.exists(rfrun_path + '/bestweight_job.h5'):
        print('loading weights')
        model.load_weights(rfrun_path + '/bestweight_job.h5')
        second_run = True

    else:
        history = model.fit(x=xtrain, y=ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[earlystop, saveBestModel, csv_logger], shuffle=True, use_multiprocessing=False,
                            validation_data=(xval, yval))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(rfrun_path + "train_val_loss.png")
        plt.show()

    print("Finished")

    
    model.load_weights(rfrun_path + '/bestweight_job.h5')
    print("load best weights")

    xtrain = []
    ytrain = []
    

    pval = model.predict(xval)
    
    np.save(rfrun_path + "/pval.npy", pval)
    auc_val, cm_val = evaluate_performance(yval, pval)
    print("auc_val", auc_val)
    print(cm_val)

    
    if modeltype == "single_input_Lasso_ge":
        xtest, ytest = get_data_GE(datapath, 'test', fold)
    if (modeltype == "single_input_Lasso_me" )|(modeltype == "Gene_me"):
        xtest, ytest = get_data_ME(datapath, 'test', fold)
    


    ptest = model.predict(xtest)
    np.save(rfrun_path + "/ptest.npy", ptest)
    auc_test, cm_test = evaluate_performance(ytest, ptest)
    print("auc_test", auc_test)
    print(cm_test)

    np.save(rfrun_path + "/auc_val.npy", auc_val)
    np.save(rfrun_path + "/auc_test.npy", auc_test)

    data = {'Pheno': str(pheno_name),
            'Omics': "ME + GE",
            'ID': [jobid],
            'fold': [fold],
            'Model': str(modeltype),
            'Val': [auc_val],
            'Test': [auc_test],
            'lr_opt': [lr_opt],
            'batch_size': [batch_size],
            'l1_value': [l1_value]}
    pd_summary_row = pd.DataFrame(data)
    pd_summary_row.to_csv(rfrun_path + "/pd_summary_row.csv")


 

    np.save(rfrun_path + "/auc_val.npy", auc_val)

    
    do_inference_all = False
    
    setnames = ['train', 'val', 'test']
    


    del model
    model = []
    gc.collect()
    tf.keras.backend.clear_session()

    return auc_val, auc_test


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "-j",
        type=int,
    )
    CLI.add_argument(
        "-lr",
        type=float,
        default=0.0005,
    )
    CLI.add_argument(
        "-bs",
        type=int,
        default=32,
    )
    CLI.add_argument(
        "-l1",
        type=float,
        default=0.01,
    )
    CLI.add_argument(
        "-mt",
        type=str,
        default="sparse_directed_gene"
    )
    CLI.add_argument(
        "-pn",
        type=str,
        default="bipolar_ukbb"
    )
    CLI.add_argument(
        "-fold",
        type=int,
        default=0
    )

    args = CLI.parse_args()
    # access CLI options
    print("jobid: " + str(args.j))

    main(jobid=args.j, lr_opt=args.lr, batch_size=args.bs, l1_value=args.l1, modeltype=args.mt,
         pheno_name=args.pn, fold=args.fold)
