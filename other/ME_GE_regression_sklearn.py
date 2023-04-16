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
import tables
import gc
import scipy
import sklearn.metrics as skm
import sklearn.linear_model as skl
import argparse
print("start")
import scipy
import sklearn.metrics as skm
from sklearn.metrics import explained_variance_score
import seaborn as sns
from sklearn.metrics import mean_squared_error

from models.RegressionModels import *
from GenNet.GenNet_utils.Utility_functions import evaluate_performance_regression
import GenNet.GenNet_utils.LocallyDirectedConnected_tf2 as LocallyDirectedConnected
from Dataloader import get_data_GE, get_data_ME



def main(jobid, lr_opt, batch_size, l1_value, modeltype, pheno_name, fold):
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


    folder = ("Results_sklearn_" + str(gt_name) + "__" + str(jobid) + "_fold_" + str(fold))
    
    rfrun_path = "/trinity/home/avanhilten/repositories/multi-omics/bios/results/MEGE_" + folder + "/"
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
    
    with open(rfrun_path + '/experiment_stats_results_.txt', 'a') as f:
        f.write('gtname = ' + str(gt_name))
        f.write('\n jobid = ' + str(jobid))
        f.write('\n model = ' + str(modeltype))
        f.write('\n folder = ' + str(folder))
        f.write('\n fold = ' + str(fold))
    
    ############ ---  Gene Expressio --- ############
    xtrain_GE, ytrain = get_data_GE(datapath, 'train', fold)
    xval_GE, yval = get_data_GE(datapath, "val", fold)
    xtest_GE, ytest = get_data_GE(datapath, 'test', fold)
    
    model = skl.Lasso(alpha=l1_value)
    #LogisticRegression(penalty='l1', solver='liblinear', n_jobs = 10, C=l1_value) 
    model.fit(xtrain_GE, ytrain)   
        
    pval = model.predict(xval_GE)
    np.save(rfrun_path + "/pval_ge.npy", pval)
    mse_val_GE, explained_variance_val_GE = evaluate_performance(yval, pval)
    print("mse_val ge ", mse_val_GE)
    print("explained_variance_val ge ", explained_variance_val_GE)

    ptest = model.predict(xtest_GE)
    np.save(rfrun_path + "/ptest_ge.npy", ptest)
    mse_test_GE, explained_variance_test_GE = evaluate_performance(ytest, ptest)
    print("mse_test me ", mse_test_GE)
    print("explained_variance_test me ", explained_variance_test_GE)


    ############ ---  Methylation --- ############
    xtrain_ME, _ = get_data_ME(datapath, 'train', fold)
    xval_ME, _ = get_data_ME(datapath, "val", fold)   
    xtest_ME, _ = get_data_ME(datapath, 'test', fold) 
    
    
    model = skl.Lasso(alpha=l1_value)
    model.fit(xtrain_ME, ytrain)   
        
    pval = model.predict(xval_ME)
    np.save(rfrun_path + "/pval_me.npy", pval)
    mse_val_ME, explained_variance_val_ME = evaluate_performance(yval, pval)
    print("mse_val_ME ", mse_val_ME)
    print("explained_variance_val_ME me ", explained_variance_val_ME)

    ptest = model.predict(xtest_ME)
    np.save(rfrun_path + "/ptest_me.npy", ptest)
    mse_test_ME, explained_variance_test_ME = evaluate_performance(ytest, ptest)
    print("mse_test me ", mse_test_ME)
    print("explained_variance_test_ME me ", explained_variance_test_ME)
    
    print(rfrun_path)

    data = {'Pheno': str(pheno_name),
            'Omics': "ME + GE",
            'ID': [jobid],
            'fold': [fold],
            'Model': str(modeltype),
            'Val_expl_GE': [explained_variance_val_GE],
            'Test_expl_GE': [explained_variance_test_GE],
            'Val_mse_GE': [mse_val_GE],
            'Test_mse_GE': [mse_test_GE],
            'Val_expl_ME': [explained_variance_val_ME],
            'Test_expl_ME': [explained_variance_test_ME],
            'Val_mse_ME': [mse_val_ME],
            'Test_mse_ME': [mse_test_ME],
            'lr_opt': [lr_opt],
            'batch_size': [batch_size],
            'rfrun_path': [rfrun_path],
            'l1_value': [l1_value]}
    
    pd_summary_row = pd.DataFrame(data)
    pd_summary_row.to_csv(rfrun_path + "/pd_summary_row.csv")

    return 

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
