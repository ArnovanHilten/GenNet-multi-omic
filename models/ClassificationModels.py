import os
import time
try:
    print('SlURM_JOB_ID',os.environ["SLURM_JOB_ID"])
except:
    print("no slurm id")
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import tensorflow as tf
import tables
import tensorflow.keras as K
import gc
import GenNet.GenNet_utils.LocallyDirectedConnected_tf2 as LocallyDirectedConnected
import scipy
import sklearn.metrics as skm
from sklearn.metrics import explained_variance_score
import seaborn as sns
from sklearn.metrics import mean_squared_error

tf.keras.backend.set_epsilon(0.0000001)

def GenNet_classification_pathway_ll_cov(inputsize_GE, inputsize_ME, inputsize_cov, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')
    
    gene_local_mask = scipy.sparse.load_npz(datapath + '/mask_gene_local.npz')
    local_mid_mask = scipy.sparse.load_npz(datapath + '/mask_local_mid.npz')
    mid_high_mask = scipy.sparse.load_npz(datapath + '/mask_mid_global.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)
    input_cov = K.Input(inputsize_cov)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer", 
                                                            kernel_regularizer=tf.keras.regularizers.l1(l1_value))(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)
    
    pathway1_layer = LocallyDirectedConnected.LocallyDirected1D(mask=gene_local_mask, filters=1, name="pathway1_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer)
    pathway1_layer = K.layers.Activation("tanh", name="activation_pathway1")(pathway1_layer)
    pathway1_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway1")(pathway1_layer)  
    
    pathway2_layer = LocallyDirectedConnected.LocallyDirected1D(mask=local_mid_mask, filters=1, name="pathway2_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway1_layer)
    pathway2_layer = K.layers.Activation("tanh", name="activation_pathway2")(pathway2_layer)
    pathway2_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway2")(pathway2_layer)   
    
    pathway3_layer = LocallyDirectedConnected.LocallyDirected1D(mask=mid_high_mask, filters=1, name="pathway3_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway2_layer)
    pathway3_layer = K.layers.Activation("tanh", name="activation_pathway3")(pathway3_layer)
    pathway3_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway3")(pathway3_layer)   

    combined_skip_nodes = K.layers.concatenate([gene_layer, pathway1_layer, pathway2_layer, pathway3_layer], axis=1)
    combined_skip_nodes = K.layers.Flatten()(combined_skip_nodes)
    
    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(combined_skip_nodes)
    end_node = K.layers.Activation("tanh", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    cov_layer = K.layers.concatenate([end_node, input_cov], axis=1)
    cov_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_end")(cov_layer)
    
    final_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                                name="final_node")(cov_layer)
    final_node = K.layers.Activation("sigmoid", name="activation_final")(final_node)
    
    model = K.Model(inputs=[input_GE, input_ME, input_cov], outputs=final_node)
    return model 

   
def GenNet_classification_pathway(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')
    
    gene_local_mask = scipy.sparse.load_npz(datapath + '/mask_gene_local.npz')
    local_mid_mask = scipy.sparse.load_npz(datapath + '/mask_local_mid.npz')
    mid_high_mask = scipy.sparse.load_npz(datapath + '/mask_mid_global.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               )(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer", 
                                                            kernel_regularizer=tf.keras.regularizers.l1(l1_value))(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)
    
    pathway1_layer = LocallyDirectedConnected.LocallyDirected1D(mask=gene_local_mask, filters=1, name="pathway1_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer)
    pathway1_layer = K.layers.Activation("tanh", name="activation_pathway1")(pathway1_layer)
    pathway1_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway1")(pathway1_layer)  
    
    pathway2_layer = LocallyDirectedConnected.LocallyDirected1D(mask=local_mid_mask, filters=1, name="pathway2_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway1_layer)
    pathway2_layer = K.layers.Activation("tanh", name="activation_pathway2")(pathway2_layer)
    pathway2_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway2")(pathway2_layer)   
    
    pathway3_layer = LocallyDirectedConnected.LocallyDirected1D(mask=mid_high_mask, filters=1, name="pathway3_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway2_layer)
    pathway3_layer = K.layers.Activation("tanh", name="activation_pathway3")(pathway3_layer)
    pathway3_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway3")(pathway3_layer)   

    combined_skip_nodes = K.layers.concatenate([gene_layer, pathway3_layer], axis=1)
    combined_skip_nodes = K.layers.Flatten()(combined_skip_nodes)
    
    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(combined_skip_nodes)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)
    
    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model     


def GenNet_classification_pathway_only(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')
    
    gene_local_mask = scipy.sparse.load_npz(datapath + '/mask_gene_local.npz')
    local_mid_mask = scipy.sparse.load_npz(datapath + '/mask_local_mid.npz')
    mid_high_mask = scipy.sparse.load_npz(datapath + '/mask_mid_global.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               )(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer", 
                                                            kernel_regularizer=tf.keras.regularizers.l1(l1_value))(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)
    
    pathway1_layer = LocallyDirectedConnected.LocallyDirected1D(mask=gene_local_mask, filters=1, name="pathway1_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer)
    pathway1_layer = K.layers.Activation("tanh", name="activation_pathway1")(pathway1_layer)
    pathway1_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway1")(pathway1_layer)  
    
    pathway2_layer = LocallyDirectedConnected.LocallyDirected1D(mask=local_mid_mask, filters=1, name="pathway2_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway1_layer)
    pathway2_layer = K.layers.Activation("tanh", name="activation_pathway2")(pathway2_layer)
    pathway2_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway2")(pathway2_layer)   
    
    pathway3_layer = LocallyDirectedConnected.LocallyDirected1D(mask=mid_high_mask, filters=1, name="pathway3_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(l1_value))(pathway2_layer)
    pathway3_layer = K.layers.Activation("tanh", name="activation_pathway3")(pathway3_layer)
    pathway3_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway3")(pathway3_layer)   

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(pathway3_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)
    
    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model     

def GenNet_classification_pathway_dense1(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')
    
    gene_local_mask = scipy.sparse.load_npz(datapath + '/mask_gene_local.npz')
    local_mid_mask = scipy.sparse.load_npz(datapath + '/mask_local_mid.npz')
    mid_high_mask = scipy.sparse.load_npz(datapath + '/mask_mid_global.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               kernel_regularizer=tf.keras.regularizers.l1(1e-8))(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer", 
                                                            kernel_regularizer=tf.keras.regularizers.l1(1e-8))(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)
    
    skip_connection = K.layers.Flatten()(gene_layer)
    skip_connection = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="skip_node")(skip_connection)  
    skip_connection = K.layers.Activation("sigmoid", name="activation_skip")(skip_connection)
    
    pathway1_layer = LocallyDirectedConnected.LocallyDirected1D(mask=gene_local_mask, filters=1, name="pathway1_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(1e-8))(gene_layer)
    pathway1_layer = K.layers.Activation("tanh", name="activation_pathway1")(pathway1_layer)
    pathway1_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway1")(pathway1_layer)  
    
    pathway2_layer = LocallyDirectedConnected.LocallyDirected1D(mask=local_mid_mask, filters=1, name="pathway2_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(1e-8))(pathway1_layer)
    pathway2_layer = K.layers.Activation("tanh", name="activation_pathway2")(pathway2_layer)
    pathway2_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway2")(pathway2_layer)   
    
    pathway3_layer = LocallyDirectedConnected.LocallyDirected1D(mask=mid_high_mask, filters=1, name="pathway3_layer", 
                                                                kernel_regularizer=tf.keras.regularizers.l1(1e-8))(pathway2_layer)
    pathway3_layer = K.layers.Activation("tanh", name="activation_pathway3")(pathway3_layer)
    pathway3_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_pathway3")(pathway3_layer)   
    pathway3_layer = K.layers.Flatten()(pathway3_layer)
    
    combined_skip_nodes = K.layers.concatenate([skip_connection, pathway3_layer], axis=1)
  
    
    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(1e-8),
                              name="end_node")(combined_skip_nodes)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)
    
    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model     


def Lasso_me(inputsize_GE, inputsize_ME, l1_value):

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    input_me_r = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    input_me_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_me_r)
    input_me_r = K.layers.Flatten()(input_me_r)

    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(input_me_r)
    end_node_act = K.layers.Activation("sigmoid", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node_act)
    return model    

   
def Lasso_ge(inputsize_GE, inputsize_ME, l1_value):


    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    input_ge_r = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    input_ge_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_ge_r)
    input_ge_r = K.layers.Flatten()(input_ge_r)
    
    
    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(input_ge_r)
    end_node_act = K.layers.Activation("sigmoid", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node_act)
    return model 
    
def GenNet_classification_combi_cov(inputsize_GE, inputsize_ME, inputsize_cov, l1_value):
    
    coord = np.eye(inputsize_GE + inputsize_cov)
    coord[:, -inputsize_cov:] = 1
    cov_mask = scipy.sparse.coo_matrix(coord)
    coord = []
    
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)
    input_cov = K.Input(inputsize_cov)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    input_cov_reshape = K.layers.Reshape(input_shape=(inputsize_cov,), target_shape=(inputsize_cov, 1))(input_cov)
    combined_cov = K.layers.concatenate([gene_layer, input_cov_reshape], axis=1)
    combined_cov = K.layers.Activation("tanh")(combined_cov)
    combined_cov = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined_cov")(combined_cov)

    gene_layer_cov = LocallyDirectedConnected.LocallyDirected1D(mask=cov_mask, filters=1,
                                                                name="gene_layer_cov")(combined_cov)
    gene_layer_cov = K.layers.Activation("tanh")(gene_layer_cov)

    gene_layer_cov = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_cov")(gene_layer_cov)

    end_node = K.layers.Flatten()(gene_layer_cov)
    end_node = K.layers.Dense(units=1, name="end_node", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(end_node)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)

    model = K.Model(inputs=[input_GE, input_ME, input_cov], outputs=end_node)
    return model
    
def GenNet_classification_combi_l_cov(inputsize_GE, inputsize_ME, inputsize_cov, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)
    input_cov = K.Input(inputsize_cov)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    
    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("tanh", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    cov_layer = K.layers.concatenate([end_node, input_cov], axis=1)
    cov_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_end")(cov_layer)
    
    final_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="final_node")(cov_layer)
    final_node = K.layers.Activation("sigmoid", name="activation_final")(final_node)
    
    model = K.Model(inputs=[input_GE, input_ME, input_cov], outputs=final_node)
    return model 


def GenNet_classification_combi(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, name="end_node", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model


def GenNet_classification_combi_l(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model

def GenNet_classification_combi_lll(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               kernel_regularizer=tf.keras.regularizers.l1(1e-8))(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer",
                                                           kernel_regularizer=tf.keras.regularizers.l1(1e-8))(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model

class omic_regularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength, begin, end):
        self.strength = strength
        self.begin = begin
        self.end = end

    def __call__(self, x):
        x2 = x[self.begin:self.end]
        return self.strength * tf.reduce_sum(tf.square(x2))

    def get_config(self):
        return {'strength': self.strength, 'begin': self.begin, 'end': self.end}


def GenNet_classification_combi_l_meth(inputsize_GE, inputsize_ME, l1_value):
    print('l1value_omic',l1value_omic)
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, 
                                                            filters=1, name="gene_layer", 
                                                            kernel_regularizer=omic_regularizer(l1value_omic, 0, 10404))(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model



def GenNet_classification_combi_l_ge(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, 
                                                            filters=1, name="gene_layer", 
                                                            kernel_regularizer=omic_regularizer(l1value_omic, 10404, 20808))(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model

def GenNet_classification_combi_cov2_ll(inputsize_GE, inputsize_ME, inputsize_cov, l1_value):
    
    coord = np.eye(inputsize_GE + inputsize_cov)
    coord[:, -inputsize_cov:] = 1
    cov_mask = scipy.sparse.coo_matrix(coord)
    coord = []
    
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)
    input_cov = K.Input(inputsize_cov)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me",
                                                               kernel_regularizer=tf.keras.regularizers.l1(l1_value))(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    input_cov_reshape = K.layers.Reshape(input_shape=(inputsize_cov,), target_shape=(inputsize_cov, 1))(input_cov)
    combined_cov = K.layers.concatenate([gene_layer, input_cov_reshape], axis=1)
    combined_cov = K.layers.Activation("tanh")(combined_cov)
    combined_cov = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined_cov")(combined_cov)

    gene_layer_cov = LocallyDirectedConnected.LocallyDirected1D(mask=cov_mask, filters=1,
                                                                name="gene_layer_cov")(combined_cov)
    gene_layer_cov = K.layers.Activation("tanh")(gene_layer_cov)
    gene_layer_cov = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_cov")(gene_layer_cov)

    
    end_node = K.layers.Flatten()(gene_layer_cov)
    end_node = K.layers.Dense(units=1, name="end_node", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(end_node)
    end_node = K.layers.Activation("tanh", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)
    
    covariate_layer = K.layers.concatenate([end_node, input_cov], axis=1)
    covariate_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_end")(covariate_layer)
    
    final_node = K.layers.Dense(units=1, 
                                kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                                name="final_node")(covariate_layer)
    final_node = K.layers.Activation("sigmoid", name="activation_final")(final_node)

    model = K.Model(inputs=[input_GE, input_ME, input_cov], outputs=final_node)
    return model


def GenNet_classification_combi_al(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              activity_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model


def GenNet_classification_combi_ll(inputsize_GE, inputsize_ME, l1_value):
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1, 
                                                               kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)

    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    end_node = K.layers.Dense(units=1,
                              kernel_regularizer=tf.keras.regularizers.l1(l1_value),
                              name="end_node")(gene_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model

def GenNet_classification_deep_5_cov(inputsize_GE, inputsize_ME, inputsize_cov, l1_value):
    
    coord = np.eye(inputsize_GE + inputsize_cov)
    coord[:, -inputsize_cov:] = 1
    cov_mask = scipy.sparse.coo_matrix(coord)
    coord = []
    
    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)
    input_cov = K.Input(inputsize_cov)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    d1_layer = K.layers.Flatten()(gene_layer)
    d1_layer = K.layers.Dense(units=100, name="d1_node")(d1_layer)
    d1_layer = K.layers.Activation("tanh", name="d1_act")(d1_layer)
    
    combined_cov = K.layers.concatenate([d1_layer, input_cov], axis=1)
    combined_cov = K.layers.Activation("tanh")(combined_cov)
    combined_cov = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined_cov")(combined_cov)
    
    d2_layer = K.layers.Dense(units=100, name="d2_node")(combined_cov)
    d2_layer = K.layers.Activation("tanh", name="d2_act")(d2_layer)
    
    d3_layer = K.layers.Dense(units=100, name="d3_node")(d2_layer)
    d3_layer = K.layers.Activation("tanh", name="d3_act")(d3_layer)
    
    d4_layer = K.layers.Dense(units=100, name="d4_node")(d3_layer)
    d4_layer = K.layers.Activation("tanh", name="d4_act")(d4_layer)
    
    d5_layer = K.layers.Dense(units=100, name="d5_node")(d4_layer)
    d5_layer = K.layers.Activation("tanh", name="d5_act")(d5_layer)

    end_node = K.layers.Dense(units=1, name="end_node")(d4_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)

    model = K.Model(inputs=[input_GE, input_ME, input_cov], outputs=end_node)
    return model
    
    
def GenNet_classification_deep_5(inputsize_GE, inputsize_ME, l1_value):
    

    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    combine_mask = scipy.sparse.load_npz(datapath + '/ME_GE_gene.npz')

    input_GE = K.Input(inputsize_GE)
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    input_GE_reshape = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    combined = K.layers.concatenate([gene_layer_ME, input_GE_reshape], axis=1)
    combined = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_combined")(combined)

    gene_layer = LocallyDirectedConnected.LocallyDirected1D(mask=combine_mask, filters=1, name="gene_layer")(combined)
    gene_layer = K.layers.Flatten()(gene_layer)
    gene_layer = K.layers.Activation("tanh", name="activation_ME_GE")(gene_layer)
    gene_layer = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(gene_layer)

    d1_layer = K.layers.Dense(units=321, name="d1_node")(gene_layer)
    d1_layer = K.layers.Activation("tanh", name="d1_act")(d1_layer)
    
    d2_layer = K.layers.Dense(units=44, name="d2_node")(d1_layer)
    d2_layer = K.layers.Activation("tanh", name="d2_act")(d2_layer)
    
    d3_layer = K.layers.Dense(units=6, name="d3_node")(d2_layer)
    d3_layer = K.layers.Activation("tanh", name="d3_act")(d3_layer)

    end_node = K.layers.Dense(units=1, name="end_node")(d3_layer)
    end_node = K.layers.Activation("sigmoid", name="activation_end")(end_node)
    end_node = K.layers.Flatten()(end_node)

    model = K.Model(inputs=[input_GE, input_ME], outputs=end_node)
    return model


def single_input_Gene_me(inputsize_ME, l1_value):

    mask_meth = scipy.sparse.load_npz(datapath + '/ME_gene.npz')
    input_ME = K.Input(inputsize_ME)

    gene_layer_ME = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    gene_layer_ME = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(gene_layer_ME)
    gene_layer_ME = LocallyDirectedConnected.LocallyDirected1D(mask=mask_meth, filters=1,
                                                               input_shape=(inputsize_ME, 1), name="gene_layer_me")(gene_layer_ME)
    gene_layer_ME = K.layers.Activation("tanh", name="activation_ME")(gene_layer_ME)

    end_node_flaten = K.layers.Flatten()(gene_layer_ME)
    end_node_bn = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene")(end_node_flaten)
    
    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(end_node_bn)
    end_node_act = K.layers.Activation("sigmoid", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=input_ME, outputs=end_node_act)
    return model    
    
    
    
def single_input_Lasso_me(inputsize_ME, l1_value):

    input_ME = K.Input(inputsize_ME)

    input_me_r = K.layers.Reshape(input_shape=(inputsize_ME,), target_shape=(inputsize_ME, 1))(input_ME)
    input_me_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_me_r)
    input_me_r = K.layers.Flatten()(input_me_r)

    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(input_me_r)
    end_node_act = K.layers.Activation("sigmoid", name="activation_end")(end_node_dense)
    

    model = K.Model(inputs=input_ME, outputs=end_node_act)
    return model    
   
def single_input_Lasso_ge(inputsize_GE, l1_value):

    input_GE = K.Input(inputsize_GE)

    input_ge_r = K.layers.Reshape(input_shape=(inputsize_GE,), target_shape=(inputsize_GE, 1))(input_GE)
    input_ge_r = K.layers.BatchNormalization(center=False, scale=False, name="inter_out_gene_layer_me")(input_ge_r)
    input_ge_r = K.layers.Flatten()(input_ge_r)
    
    
    end_node_dense = K.layers.Dense(units=1, name="end_node_dense", kernel_regularizer=tf.keras.regularizers.l1(l1_value))(input_ge_r)
    end_node_act = K.layers.Activation("sigmoid", name="activation_end")(end_node_dense)
    
    model = K.Model(inputs=input_GE, outputs=end_node_act)
    return model 


