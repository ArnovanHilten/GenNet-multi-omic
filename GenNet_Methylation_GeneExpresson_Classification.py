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
import sklearn.metrics as skm

tf.keras.backend.set_epsilon(0.0000001)

from models.ClassificationModels import *
from GenNet.GenNet_utils.Utility_functions import evaluate_performance
import GenNet.GenNet_utils.LocallyDirectedConnected_tf2 as LocallyDirectedConnected
from Dataloader import get_data_GE, get_data_ME

print("start")


def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(y_true, 0.0001, 1)
    y_pred = K.backend.clip(y_pred, 0.0001, 1)

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_possitive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)


def main(jobid, lr_opt, batch_size, l1_value, modeltype, pheno_name, fold, omic_l1):
    print('tensorflowversion:', tf.__version__)
    global gt_name, datapath, weight_possitive_class, weight_negative_class, l1value_omic
    
    ##
    # Change the paths here 
    datapath = "/trinity/home/avanhilten/repositories/multi-omics/bios/processed_data/"
    resultpath = "./results/"
    ##
    
    jobid = int(jobid)
    lr_opt = float(lr_opt)
    batch_size = int(batch_size)
    l1_value = float(l1_value)
    fold = int(fold)
    epochs = 2000
    dropout = 0
    extraname = ""
    augment = False
    gt_name = pheno_name
    weight_negative_class = 1
    second_run = False
    wpc = 1
    l1value_omic  = omic_l1

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
        weight_possitive_class =  (ytrain.shape[0] - ytrain["labels"].sum()) / ytrain["labels"].sum()
    else:
        weight_possitive_class = wpc

    xtrain_ME, _ = get_data_ME(datapath, 'train', fold)
    xtrain_GE, ytrain = get_data_GE(datapath, 'train', fold)
    xtrain = [xtrain_GE, xtrain_ME]

    xval_GE, yval = get_data_GE(datapath, "val", fold)
    xval_ME, _ = get_data_ME(datapath, "val", fold)
    xval = [xval_GE, xval_ME]

    folder = ("Results_" + str(gt_name) + "__" + str(jobid) + "_fold_" + str(fold))

    inputsize_GE = xtrain_GE.shape[1]
    inputsize_ME = xtrain_ME.shape[1]
    
    rfrun_path = resultpath + folder + "/"
    if not os.path.exists(rfrun_path):
        print("Runpath did not exist but is made now")
        os.mkdir(rfrun_path)

    try:    
        with open(rfrun_path + '/Slurm_'+str(os.environ["SLURM_JOB_ID"])+'_.txt', 'w') as f:
            f.write('gtname = ' + str(os.environ["SLURM_JOB_ID"]))
    except:
        print("no slurm job id")
        
    print("jobid =  " + str(jobid))
    print("fold =  " + str(fold))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))

    
    if modeltype == "GenNet_classification_pathway_dense1":
        model = GenNet_classification_pathway_dense1(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)     
    if modeltype == "GenNet_classification_combi_lll":
        model = GenNet_classification_combi_lll(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)
    
    if modeltype == "GenNet_classification_pathway":
        model = GenNet_classification_pathway(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)  
    if modeltype == "GenNet_classification_pathway_only":
        model = GenNet_classification_pathway_only(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value) 
    if modeltype == "GenNet_classification_deep_5":
        model = GenNet_classification_deep_5(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)  
    
    if modeltype == "GenNet_classification_combi_l_meth":
        model = GenNet_classification_combi_l_meth(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value) 
        
    if modeltype == "GenNet_classification_combi_l_ge":
        model = GenNet_classification_combi_l_ge(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)          
    
    if modeltype == "GenNet_classification_combi":
        model = GenNet_classification_combi(inputsize_GE=int(inputsize_GE),
                                            inputsize_ME=int(inputsize_ME),
                                            l1_value=l1_value)
    if modeltype == "GenNet_classification_combi_l":
        model = GenNet_classification_combi_l(inputsize_GE=int(inputsize_GE),
                                              inputsize_ME=int(inputsize_ME),
                                              l1_value=l1_value)
        
    if modeltype == "GenNet_classification_combi_ll":
        model = GenNet_classification_combi_ll(inputsize_GE=int(inputsize_GE),
                                              inputsize_ME=int(inputsize_ME),
                                              l1_value=l1_value)
    if modeltype == "GenNet_classification_combi_al":
        model = GenNet_classification_combi_al(inputsize_GE=int(inputsize_GE),
                                               inputsize_ME=int(inputsize_ME),
                                               l1_value=l1_value)
        
        
    if "_cov" in modeltype:
        covariates_train = pd.read_csv(datapath + "ytrain_" + gt_name + "_"+str(fold)+".csv")[["DNA_BloodSampling_Age","Sex"]]
        covariates_train = ((covariates_train - covariates_train.mean()) / covariates_train.std()).values
        covariates_val =   pd.read_csv(datapath + "yval_" + gt_name + "_"+str(fold)+".csv")[["DNA_BloodSampling_Age","Sex"]]
        covariates_val = ((covariates_val - covariates_val.mean()) / covariates_val.std()).values
        inputsize_cov = 2
        
        xtrain  = [xtrain_GE, xtrain_ME, covariates_train]
        
        print(xtrain_GE.shape, xtrain_ME.shape, covariates_train.shape  )
        xval = [xval_GE, xval_ME, covariates_val]
        print(xval_GE.shape, xval_ME.shape, covariates_val.shape  )
        
    
        if (modeltype == "GenNet_classification_combi_cov"):
            model = GenNet_classification_combi_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)
        if (modeltype == "GenNet_classification_combi_l_cov"):
            model = GenNet_classification_combi_l_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)   
            
        if (modeltype == "GenNet_classification_combi_cov2_ll"):
            model = GenNet_classification_combi_cov2_ll(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)           
        if (modeltype == "GenNet_classification_pathway_ll_cov"):
            model = GenNet_classification_pathway_ll_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value) 
        if (modeltype == "GenNet_classification_deep_5_cov"):
            model = GenNet_classification_deep_5_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)             
            

    if modeltype == "Lasso_ge":
        model = Lasso_ge(inputsize_GE=int(inputsize_GE),
                                          inputsize_ME=int(inputsize_ME),
                                          l1_value=l1_value)
        
    if modeltype == "Lasso_me":
        model = Lasso_me(inputsize_GE=int(inputsize_GE),
                                          inputsize_ME=int(inputsize_ME),
                                          l1_value=l1_value)    

    xtrain_ME = []
    xtrain_GE = []
    xval_ME = []
    xval_GE = []
    
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
                            callbacks=[earlystop, saveBestModel, csv_logger], shuffle=True, use_multiprocessing=True, workers=3,
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
    
    check_performance_train = True
    if check_performance_train & second_run:
        ptrain = model.predict(xtrain).flatten()
        np.save(rfrun_path + "/ptrain.npy", ptrain)
        auc_train, cm_train = evaluate_performance(ytrain, ptrain)
        print("auc_train", auc_train)
        print(cm_train)

    xtrain = []
    ytrain = []
    gc.collect()
    
#     ptrain = model.predict(xtrain)
#     np.save(rfrun_path + "/ptrain.npy", ptrain)

#     auc_train, cm_train = evaluate_performance(ytrain, ptrain)
#     print("auc_train", auc_train)
#     print(auc_train)

    pval = model.predict(xval)
    np.save(rfrun_path + "/pval.npy", pval)
    auc_val, cm_val = evaluate_performance(yval, pval)
    print("auc_val", auc_val)
    print(cm_val)

    xtest_GE, ytest = get_data_GE(datapath, 'test', fold)
    xtest_ME, _ = get_data_ME(datapath, 'test', fold)
    xtest = [xtest_GE, xtest_ME]
    
    if "_cov" in modeltype:
        covariates_test =   pd.read_csv(datapath + "ytest_" + gt_name + "_"+str(fold)+".csv")[["DNA_BloodSampling_Age","Sex"]]
        covariates_test = ((covariates_test - covariates_test.mean()) / covariates_test.std()).values
        xtest = [xtest_GE, xtest_ME, covariates_test]

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


    checkformer = False

    if checkformer:
        xformer_ME, _ = get_data_ME(datapath, 'former_smoker')
        xformer_GE, yformer = get_data_GE(datapath, 'former_smoker')
        xformer = [xformer_GE, xformer_ME]

        pformer = model.predict(xformer)
        np.save(rfrun_path + "/pformer.npy", pformer)
        setnames.append('xformer')
        evaluatesets.append(xformer)
        print(cm_test)

    np.save(rfrun_path + "/auc_val.npy", auc_val)

    
    do_inference_all = False
    
    setnames = ['train', 'val', 'test']
    
    if second_run & do_inference_all:
        print("second run")

        for setname in setnames:
            
            evaluateset_GE, evaluateset_Y = get_data_GE(datapath, setname, fold)
            evaluateset_ME, _ = get_data_ME(datapath, setname, fold)
            evaluateset = [evaluateset_GE, evaluateset_ME]
            evaluateset_GE = []
            evaluateset_ME = []
            
            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_ME').output)
            intermediate_layer_model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                                             metrics=["accuracy"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_ME" + setname + ".npy", intermediate_output)

            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_ME_GE').output)
            intermediate_layer_model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                                             metrics=["accuracy"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_ME_GE" + setname + ".npy", intermediate_output)

            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_end').output)
            intermediate_layer_model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                                             metrics=["accuracy"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_end" + setname + ".npy", intermediate_output)

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
    CLI.add_argument(
        "-omic_l1",
        type=float,
        default=0.01,
    )
    args = CLI.parse_args()
    # access CLI options
    print("jobid: " + str(args.j))

    main(jobid=args.j, lr_opt=args.lr, batch_size=args.bs, l1_value=args.l1, modeltype=args.mt,
         pheno_name=args.pn, fold=args.fold, omic_l1 = args.omic_l1)
