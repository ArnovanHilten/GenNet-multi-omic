import os
import time
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
import tables
import tensorflow.keras as K
import gc
import LocallyDirectedConnected_tf2 as LocallyDirectedConnected
import scipy
import sklearn.metrics as skm
from sklearn.metrics import explained_variance_score
import seaborn as sns
from sklearn.metrics import mean_squared_error

tf.keras.backend.set_epsilon(0.0000001)
import argparse

print("start")
from models.RegressionModels import *
from GenNet.GenNet_utils.Utility_functions import evaluate_performance_regression as evaluate_performance
import GenNet.GenNet_utils.LocallyDirectedConnected_tf2 as LocallyDirectedConnected
from Dataloader import get_data_GE, get_data_ME



def main(jobid, lr_opt, batch_size, l1_value, modeltype, pheno_name, fold, omic_l1):
    print(tf.__version__)
    global gt_name, datapath, weight_possitive_class, weight_negative_class, bias_init, l1value_omic


    datapath = "//trinity/home/avanhilten/repositories/multi-omics/bios/processed_data/"
    resultpath = "./results/"
    
    if pheno_name == "Age":
        bias_init = 52.25
        print("bias age", bias_init)
    elif pheno_name == "LDL":
        bias_init = 3.12
        print("bias LDL", bias_init)
    else:
        print("no bias defined")
        exit()


    print(jobid)
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
    second_run = False
    l1value_omic  = omic_l1

   
    if lr_opt == 0:
        optimizer = tf.keras.optimizers.Adadelta()
        print("adadelta")
    else:
        optimizer = tf.keras.optimizers.Adam(lr=lr_opt)
        print("Adam", lr_opt)

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
            f.write('slurm id= ' + str(os.environ["SLURM_JOB_ID"]))
    except:
        print("no slurm job id")
        
    print("jobid =  " + str(jobid))
    print("fold =  " + str(fold))
    print("folder = " + str(folder))
    print("batchsize = " + str(batch_size))

    
    
    if '_cov' in modeltype:
        print("covariates")
        covariates_train = pd.read_csv(datapath + "ytrain_" + gt_name + "_"+str(fold)+".csv")[["Sex"]].values
#         covariates_train = ((covariates_train - covariates_train.mean()) / covariates_train.std()).values
        covariates_val =   pd.read_csv(datapath + "yval_" + gt_name + "_"+str(fold)+".csv")[["Sex"]].values
#         covariates_val = ((covariates_val - covariates_val.mean()) / covariates_val.std()).values
        inputsize_cov = 1
        
        xtrain  = [xtrain_GE, xtrain_ME, covariates_train]
        
        print(xtrain_GE.shape )
        print(xtrain_ME.shape )
        print(covariates_train.shape)
        
        xval = [xval_GE, xval_ME, covariates_val]
        
        print(xval_GE.shape, xval_ME.shape, covariates_val.shape  )
        
      
        if (modeltype == "GenNet_regression_combi_cov"):
            model = GenNet_regression_combi_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)
        if (modeltype == "GenNet_regression_combi_l_cov"):
            model = GenNet_regression_combi_l_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)      
            
        if (modeltype == "GenNet_regression_combi_cov2_bl"):
            model = GenNet_regression_combi_cov2_bl(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)             
            
        if (modeltype == "GenNet_regression_combi_cov2_bll"):
            model = GenNet_regression_combi_cov2_bll(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value) 
        if (modeltype == "GenNet_regression_combi_cov_bll"):
            model = GenNet_regression_combi_cov_bll(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value) 
        if (modeltype == "GenNet_regression_pathway_ll_cov"):
            model = GenNet_regression_pathway_ll_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)             
        if (modeltype == "GenNet_regression_deep_cov"):
            model = GenNet_regression_deep_cov(inputsize_GE=int(inputsize_GE),
                                           inputsize_ME=int(inputsize_ME),
                                           inputsize_cov = int(inputsize_cov),
                                           l1_value=l1_value)          

    if modeltype == "GenNet_regression_pathway_dense1":
        model = GenNet_regression_pathway_dense1(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)
    if modeltype == "GenNet_regression_combi_blll":
        model = GenNet_regression_combi_blll(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)     
    if modeltype == "GenNet_regression_pathway_only":
        model = GenNet_regression_pathway_only(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)    
    if modeltype == "GenNet_regression_pathway":
        model = GenNet_regression_pathway(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)  
    if modeltype == "GenNet_regression_combi_bl_meth":
        model = GenNet_regression_combi_bl_meth(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)  
    if modeltype == "GenNet_regression_combi_bl_ge":
        model = GenNet_regression_combi_bl_ge(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)       
    if modeltype == "GenNet_regression_deep_5":
        model = GenNet_regression_deep_5(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)    
    if modeltype == "GenNet_regression_combi_bl":
        model = GenNet_regression_combi_bl(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)  
    if modeltype == "GenNet_regression_combi_bll":
        model = GenNet_regression_combi_bll(inputsize_GE=int(inputsize_GE), 
                                        inputsize_ME=int(inputsize_ME),
                                        l1_value = l1_value)          
        
    if modeltype == "Lasso_ge":
        model = Lasso_ge(inputsize_GE=int(inputsize_GE),
                                          inputsize_ME=int(inputsize_ME),
                                          l1_value=l1_value)
        
    if modeltype == "Lasso_me":
        model = Lasso_me(inputsize_GE=int(inputsize_GE),
                                          inputsize_ME=int(inputsize_ME),
                                          l1_value=l1_value)          
        
        
    xval_GE = []
    xval_ME = []
    xtrain_GE = []
    xtrain_ME = []    

    model.compile(loss="mse", optimizer=optimizer, metrics=["mse","mae"])

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
    time_start = time.time()
    if os.path.exists(rfrun_path + '/bestweight_job.h5'):
        print('loading weights')
        model.load_weights(rfrun_path + '/bestweight_job.h5')
        second_run = True

    else:
        history = model.fit(x=xtrain, y=ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[earlystop, saveBestModel, csv_logger], shuffle=True,
                            workers=1, use_multiprocessing=False,
                            validation_data=(xval, yval))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(rfrun_path + "train_val_loss.png")
        plt.show()
        
    time_end = time.time()
    time_spend = time_end - time_start
    print("Finished in", time_spend)



    model.load_weights(rfrun_path + '/bestweight_job.h5')
    print("load best weights")
    gc.collect()

    eval_train = False
    
    if eval_train:
    
        ptrain = model.predict(xtrain)
        np.save(rfrun_path + "/ptrain.npy", ptrain)
        explained_variance_train = explained_variance_score(ytrain, ptrain)
        mse_train = mean_squared_error(ytrain, ptrain)
        print("mean_squared_error train = ", mse_train)
        print("root_mean_squared_error train = ", np.sqrt(mse_train))
        print("ptrain max", ptrain.max())
        print("ptrain min", ptrain.min())
        print("ptrain mean", ptrain.mean())
        print("explained variance train", explained_variance_train)
        with open(rfrun_path + '/experiment_stats_results_.txt', 'a') as f:
            f.write("\n  ptrain max = " + str(ptrain.max()))
            f.write("\n  ptrain min = " + str(ptrain.min()))
            f.write("\n  ptrain mean = " + str(ptrain.mean()))
            f.write("\n  mean_squared_error train = " + str(mse_train))
            f.write("\n  explained_variance_train = " + str(explained_variance_train))

        plt.figure()
        sns.jointplot(ytrain, ptrain)
        plt.savefig(rfrun_path + "jointplot_train.png")
    
    xtrain = []
    ytrain = []
    gc.collect()
    
    pval = model.predict(xval)
    pval = pval.flatten()
    yval = yval.flatten()
    
    np.save(rfrun_path + "/pval.npy", pval)
    mse_val = mean_squared_error(yval, pval)
    explained_variance_val = explained_variance_score(yval, pval)
    print("mean_squared_error val = ", mse_val)
    print("root_mean_squared_error val = ", np.sqrt(mse_val))
    print("pval max", pval.max())
    print("pval min", pval.min())
    print("pval mean", pval.mean())
    print("explained variance val", explained_variance_val)
    with open(rfrun_path + '/experiment_stats_results_.txt', 'a') as f:
        f.write("\n  explained_variance_val = " + str(explained_variance_val))
        f.write("\n  pval max = " + str(pval.max()))
        f.write("\n  pval min = " + str(pval.min()))
        f.write("\n  pval mean = " + str(pval.mean()))
        f.write("\n  mean_squared_error val = " + str(mse_val))
   
    plt.figure()
    sns.jointplot(yval, pval)
    plt.savefig(rfrun_path + "jointplot_val.png")

    xtest_GE, ytest = get_data_GE(datapath, 'test', fold)
    xtest_ME, _ = get_data_ME(datapath, 'test', fold)
    xtest = [xtest_GE, xtest_ME]
    
    if '_cov' in modeltype:
        
        covariates_test =   pd.read_csv(datapath + "ytest_" + gt_name + "_"+str(fold)+".csv")[["Sex"]].values
#         covariates_test = ((covariates_test - covariates_test.mean()) / covariates_test.std()).values
        xtest = [xtest_GE, xtest_ME, covariates_test]
    
    ptest = model.predict(xtest)
    
    ptest = ptest.flatten()
    ytest = ytest.flatten()
    
    np.save(rfrun_path + "/ptest.npy", ptest)
    mse_test = mean_squared_error(ytest, ptest)
    
    
    explained_variance_test = explained_variance_score(ytest, ptest)
    print("mean_squared_error test = ", mse_test)
    print("root_mean_squared_error test = ", np.sqrt(mse_test))
    print("ptest max", ptest.max())
    print("ptest min", ptest.min())
    print("ptest mean", ptest.mean())
    print("explained variance test", explained_variance_test)
    with open(rfrun_path + '/experiment_stats_results_.txt', 'a') as f:
        f.write("\n auc explained_variance_test = " + str(explained_variance_test))
        f.write("\n  ptest max = " + str(ptest.max()))
        f.write("\n  ptest min = " + str(ptest.min()))
        f.write("\n  ptest mean = " + str(ptest.mean()))
        f.write("\n  mean_squared_error test = " + str(mse_test))
        f.write("\n auc explained_variance_test = " + str(explained_variance_test))

    plt.figure()
    sns.jointplot(ytest, ptest)
    plt.savefig(rfrun_path + "jointplot_test.png")

    
    
    np.save(rfrun_path + "/time_spend.npy", time_spend)
    np.save(rfrun_path + "/explained_variance_val.npy", explained_variance_val)
    np.save(rfrun_path + "/explained_variance_test.npy", explained_variance_test)

    data = {'Pheno': str(pheno_name),
            'Omics': "ME + GE",
            'ID': [jobid],
            'fold': [fold],
            'Model': str(modeltype),
            'Val_expl': [explained_variance_val],
            'Test_expl': [explained_variance_test],
            'Val_mse': [mse_val],
            'Test_mse': [mse_test],
            'lr_opt': [lr_opt],
            'batch_size': [batch_size],
            'l1_value': [l1_value]}
    
    pd_summary_row = pd.DataFrame(data)
    pd_summary_row.to_csv(rfrun_path + "/pd_summary_row.csv")

    evaluate_all = True
    if second_run & evaluate_all:
        
        setnames = ['train', 'val', 'test']
        print("second run")

        for setname in setnames:
            
            evaluateset_GE, evaluateset_Y = get_data_GE(datapath, setname, fold)
            evaluateset_ME, _ = get_data_ME(datapath, setname, fold)
            evaluateset = [evaluateset_GE, evaluateset_ME]
            evaluateset_GE = []
            evaluateset_ME = []
  
    
            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_ME').output)
            intermediate_layer_model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_ME" + setname + ".npy", intermediate_output)

            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_ME_GE').output)
            intermediate_layer_model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_ME_GE" + setname + ".npy", intermediate_output)

            intermediate_layer_model = K.Model(inputs=model.input,
                                               outputs=model.get_layer(name='activation_end').output)
            intermediate_layer_model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])
            intermediate_output = intermediate_layer_model.predict(evaluateset)
            np.save(rfrun_path + "/activation_end" + setname + ".npy", intermediate_output)

    del model
    model = []
    gc.collect()
    tf.keras.backend.clear_session()
    print("done")
    return mse_val, mse_test


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
         pheno_name=args.pn, fold=args.fold, omic_l1 =args.omic_l1 )
