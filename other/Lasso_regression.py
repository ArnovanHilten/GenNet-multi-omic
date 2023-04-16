import os

try:
    print('SlURM_JOB_ID',os.environ["SLURM_JOB_ID"])
except:
    print("no slurm id")
import matplotlib


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



def get_data_GE(datapath, setname, fold):
    ytest = pd.read_csv(datapath + "y" + str(setname) + "_" + gt_name + "_"+str(fold)+".csv")
    h5file = tables.open_file(datapath + "GE_CPM_overlap_transposed.h5", "r")
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


def Gene_me(inputsize_ME, l1_value):

    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("relu", name="activation_ME")(gene_layer_ME)

    end_node_flaten = K.layers.Flatten()(gene_layer_ME)
    end_node_bn = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(end_node_flaten)
    
    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(end_node_bn)
    end_node_act = K.layers.Activation("relu", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=input_ME, outputs=end_node_act)
    return model    

def Lasso_me(inputsize_ME, l1_value):

    input_ME = K.Input(inputsize_ME)

    input_me_r = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    input_me_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_me_r)
    input_me_r = K.layers.Flatten()(input_me_r)

    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", 
                                    kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                                    bias_initializer= tf.keras.initializers.Constant(bias_init)
                                    )(input_me_r)
    end_node_act = K.layers.Activation("relu", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs = input_ME, outputs=end_node_act)
    return model    
   
def Lasso_ge(inputsize_GE, l1_value):


    input_GE = K.Input(inputsize_GE)

    input_ge_r = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    input_ge_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_ge_r)
    input_ge_r = K.layers.Flatten()(input_ge_r)
    
    
    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", 
                                    kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                                    bias_initializer= tf.keras.initializers.Constant(bias_init))(input_me_r)
    end_node_act = K.layers.Activation("relu", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=input_GE, outputs=end_node_act)
    return model 


    

def evaluate_performance(y, p):
    explained_variance = explained_variance_score(y, p)
    mse = mean_squared_error(y, p)
    print("mean_squared_error train = ", mse)
    print("ptrain max", p.max())
    print("ptrain min", p.min())
    print("ptrain mean", p.mean())
    print("explained variance train", explained_variance)

    return mse, explained_variance


def main(jobid, lr_opt, batch_size, l1_value, modeltype, pheno_name, fold):
    print(tf.__version__)
    global gt_name, datapath, weight_possitive_class, weight_negative_class, bias_init


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
    datapath = "//trinity/home/avanhilten/repositories/multi-omics/bios/processed_data/"
    epochs = 2000
    dropout = 0
    extraname = ""
    augment = False
    gt_name = pheno_name
    second_run = False
    

    

    if lr_opt == 0:
        optimizer = tf.keras.optimizers.Adadelta()
        print("adadelta")
    else:
        optimizer = tf.keras.optimizers.Adam(lr=lr_opt)
        print("Adam", lr_opt)

    if modeltype == "Lasso_ge":
        xtrain, ytrain = get_data_GE(datapath, 'train', fold)
        xval, yval = get_data_GE(datapath, "val", fold)
    if (modeltype == "Lasso_me" )|(modeltype == "Gene_me"):
        xtrain, ytrain = get_data_ME(datapath, 'train', fold)
        xval, yval = get_data_ME(datapath, "val", fold)

    folder = ("Results_" + str(gt_name) + "__" + str(jobid) + "_fold_" + str(fold))

    inputsize= xtrain.shape[1]

    


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



    if modeltype == "Lasso_ge":
        model = Lasso_ge(inputsize_GE=int(inputsize), l1_value=l1_value)   
    if modeltype == "Lasso_me":
        model = Lasso_me(inputsize_ME=int(inputsize), l1_value=l1_value)    
    if modeltype == "Gene_me":
        model = Gene_me(inputsize_ME=int(inputsize), l1_value=l1_value)          
        
        
        

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
    # %%
    if os.path.exists(rfrun_path + '/bestweight_job.h5'):
        print('loading weights')
        model.load_weights(rfrun_path + '/bestweight_job.h5')
        second_run = True

    else:
        history = model.fit(x=xtrain, y=ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[earlystop, saveBestModel, csv_logger], shuffle=True,
                            workers=5, use_multiprocessing=True,
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

    pval = model.predict(xval)
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

    
    
    
    if modeltype == "Lasso_ge":
        xtest, ytest = get_data_GE(datapath, 'test', fold)
    if (modeltype == "Lasso_me" )|(modeltype == "Gene_me"):
        xtest, ytest = get_data_ME(datapath, 'test', fold)
        
    
    
    ptest = model.predict(xtest)
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

    


    np.save(rfrun_path + "/explained_variance_val.npy", explained_variance_val)
    np.save(rfrun_path + "/explained_variance_test.npy", explained_variance_test)

    data = {'Pheno': str(pheno_name),
            'Omics': "ME + GE",
            'ID': [jobid],
            'fold': [fold],
            'Model': str(modeltype),
            'Train_expl': [explained_variance_train],
            'Val_expl': [explained_variance_val],
            'Test_expl': [explained_variance_test],
            'Val_mse': [mse_test],
            'Test_mse': [mse_val],
            'Train_mse': [mse_train],
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

    args = CLI.parse_args()
    # access CLI options
    print("jobid: " + str(args.j))

    main(jobid=args.j, lr_opt=args.lr, batch_size=args.bs, l1_value=args.l1, modeltype=args.mt,
         pheno_name=args.pn, fold=args.fold)
