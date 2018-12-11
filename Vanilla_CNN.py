#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:10:14 2018

0-9: position 1-10
A-Z: position 11-36

Each small individual character will be treated as independent sample for training, and in the 
evaluation process, we will treat 7 as one

@author: jiryi
@contributor: qiqi Ke Ma
"""

import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import random


# In[]: data import and exploration
print('Data importing...')
N_class = 36 # choice of characters
C_h = 40 # character image height
C_w = 20 # character image width
I_h = 320 # car image height
I_w = 240 # car image width
L_p = 7 # plate length
N_parts = 9
N_sample = np.sum([50,60,52,50,29,60,32,92,51])# 

data_PCL = [0]*N_parts # label for each character
data_PL = [0]*N_parts # label for each plate
data_PCI = [0]*N_parts # feature data for each character
data_PI = [0]*N_parts # feature for each car image
    
for i in np.arange(N_parts):
    pcl = sio.loadmat('Plate_Character_Labels_%s.mat' % str(i))
    pcl = pcl["data_lab"] # 50,60,52,50,29,60,32,92,51
    data_PCL[i] = pcl
    
    pl = sio.loadmat('Plate_Labels_%s.mat' % str(i))
    pl = pl["data_cha"]
    data_PL[i] = pl
    
    pci = sio.loadmat('Plates_Character_Images_%s.mat' % str(i))
    pci = pci["data_feat"]
    data_PCI[i] = pci
    
    pi = sio.loadmat('Plates_Images_%s.mat' % str(i))
    pi = pi["data_img"]
    data_PI[i] = pi

data_PCL = np.concatenate(data_PCL)
data_PL = np.concatenate(data_PL)
data_PCI = np.concatenate(data_PCI)
data_PI = np.concatenate(data_PI)

# In[]
def vtl(vector):
    """JYI
    vector to label convertor,note that: 0-9 correspond to the 1st-10th position in the vector, A-Z for 11-36
    the 1st position start with index 0, and the 36th position has index 35
    """
    index = np.nonzero(vector)
    index = index[0]
    index = index[0]
    label = 0
    if index==0:
        label='0'
    elif index==1:
        label='1'
    elif index==2:
        label='2'
    elif index==3:
        label='3'
    elif index==4:
        label='4'
    elif index==5:
        label='5'
    elif index==6:
        label='6'
    elif index==7:
        label='7'
    elif index==8:
        label='8'
    elif index==9:
        label='9'
    elif index==10:
        label='A'
    elif index==11:
        label='B'
    elif index==12:
        label='C'
    elif index==13:
        label='D'
    elif index==14:
        label='E'
    elif index==15:
        label='F'
    elif index==16:
        label='G'
    elif index==17:
        label='H'
    elif index==18:
        label='I'
    elif index==19:
        label='J'
    elif index==20:
        label='K'
    elif index==21:
        label='L'
    elif index==22:
        label='M'
    elif index==23:
        label='N'
    elif index==24:
        label='O'
    elif index==25:
        label='P'
    elif index==26:
        label='Q'
    elif index==27:
        label='R'
    elif index==28:
        label='S'
    elif index==29:
        label='T'
    elif index==30:
        label='U'
    elif index==31:
        label='V'
    elif index==32:
        label='W'
    elif index==33:
        label='X'
    elif index==34:
        label='Y'
    else:
        label='Z'
        
    return label 

# In[]: sample visualization
def samp_extract(rand_ind):
    # every time process one index
    rand_samp_PCL = [0]*L_p
    rand_samp_PCI = [0]*L_p
    for i in np.arange(L_p):
        ind = (rand_ind)*L_p + i
        vector = data_PCL[ind] # no need for transpose or inverse
        # rand_samp_PCL[i] = utils.vtl(vector)
        rand_samp_PCL[i] = vtl(vector)
        rand_samp_PCI[i] = data_PCI[ind]
    
    
    rand_samp_PL = data_PL[rand_ind]
    rand_samp_PL = rand_samp_PL[0]
    rand_samp_PI = data_PI[rand_ind]
    
    # visualization of data sample
    plt.figure 
    plt.imshow(np.transpose(rand_samp_PI.reshape((I_w,I_h))))
    plt.title("Car plate reads: %s" % str(rand_samp_PL))
    plt.show()
    
    plt.figure
    for i in np.arange(L_p):
        plt.subplot(1,7,i+1)
        plt.imshow(np.transpose(rand_samp_PCI[i].reshape(((C_w,C_h)))))
        plt.title('%s' % str(rand_samp_PCL[i]))
    plt.show() 
    
# In[]: data exploration
print('Data exploration...')
mode = 1 # 1 for random checking; 0 for full checking
if mode==1:
    rand_ind = np.random.randint(0,N_sample)
    # rand_ind = 231 # first plate sample, 231 for the last 
    samp_extract(rand_ind) 
else:
    for rand_ind in np.arange(0, N_sample): # for rand_ind in np.arange(N_sample):
        samp_extract(rand_ind)
# 50,60,52,50,29,60,32,92,51
# In[]: plate image and character image indexing
def pc_index(Pind_set):
    # Pind_set is the index set of plate images (not the character images)
    # Cind_set is the index set of character images
    # set_card = len(ind_set)
    Cind_set = []
    for Pind in Pind_set:
        for j in np.arange(L_p):
            Cind = Pind*L_p + j
            Cind_set.append(Cind)
    return Cind_set

# In[]: training data and testing data separation
# The train_PCL, train_PCI, test_PCL, test_PCI are what we will use for training and evaluation        
        
Itrain_N = int(0.8*N_sample)
Itrain_ind = random.sample(range(N_sample),Itrain_N)
Itest_N = N_sample - Itrain_N
Itest_ind = np.setdiff1d(np.arange(N_sample),Itrain_ind)

# train_PL = data_PL[Itrain_ind]
# train_PI = data_PI[Itrain_ind]
# test_PL = data_PL[Itest_ind]
# test_PI = data_PI[Itest_ind]

Ctrain_N = Itrain_N*L_p
Ctest_N = Itest_N*L_p

Ctrain_ind = pc_index(Itrain_ind)
Ctest_ind = np.setdiff1d(np.arange(N_sample*L_p),Ctrain_ind)

train_PCL = data_PCL[Ctrain_ind]
train_PCL_1d = np.argmax(train_PCL,axis=1) # along all dimensions if no specified;
test_PCL = data_PCL[Ctest_ind]
test_PCL_1d = np.argmax(test_PCL,axis=1)
# test_PCL_lab_ind = np.argmax(test_PCL,axis=1) # label index

train_PCI = data_PCI[Ctrain_ind]/255.0 # scale down
train_PCI = np.reshape(train_PCI,(Ctrain_N,C_w,C_h))
train_PCI = np.transpose(train_PCI,(0,2,1))
test_PCI = data_PCI[Ctest_ind]/255.0
test_PCI = np.reshape(test_PCI,(Ctest_N,C_w,C_h))
test_PCI = np.transpose(test_PCI,(0,2,1))

# In[]: neural network hyper-parameters
l_rate = 0.005
batch_size = 90 # number of plates; batch_size*L_p characters
# MaxIte = 500
N_epoch = 500

conv1_fsize = 4
conv1_fnum = 20
conv2_fsize = 4
conv2_fnum = 20
conv3_fsize = 4
conv3_fnum = 20

maxpool1_ksize = 2
maxpool2_ksize = 2
maxpool3_ksize = 2

fc1 = 100
fc2 = 36

# In[]: cnn + maxpooling 
def cnn_maxpool():
    a = 0
    
    # maxpooling: take BHWC input, 1HW1 kernel, 1HW1 stride 
    return a

# In[]: neural network construction, convolutional layers
# VALID: without padding; SAME: with padding
# batch_pci_ext, 4d array
# batch_pcl, numerical value for indicating class (1d); from train_PCL_1d or test_PCL_1d

batch_pcl = tf.placeholder(shape=[batch_size*L_p],dtype=tf.int32)
batch_pci = tf.placeholder(shape=(batch_size*L_p,C_h,C_w),dtype=tf.float32)
batch_pci_4d = tf.expand_dims(batch_pci,axis=3) # 4d array for convolution operation,BHWC

# conv1 + relu + maxpool1
conv1_filter = tf.Variable(tf.truncated_normal([conv1_fsize,conv1_fsize,1,conv1_fnum],
                                               stddev=0.1,dtype=tf.float32)) # HWIO 
conv1_bias = tf.Variable(tf.truncated_normal([conv1_fnum],stddev=0.1,dtype=tf.float32))
conv1_output = tf.nn.conv2d(batch_pci_4d,conv1_filter,
                            strides=[1,1,1,1],padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1_output,conv1_bias))
maxpool1 = tf.nn.max_pool(relu1, ksize=[1, maxpool1_ksize, maxpool1_ksize, 1],
                           strides=[1, maxpool1_ksize, maxpool1_ksize, 1], padding='SAME')

# conv2 + relu + maxpool2
conv2_filter = tf.Variable(tf.truncated_normal([conv2_fsize,conv2_fsize,conv1_fnum,conv2_fnum],
                                               stddev=0.1,dtype=tf.float32))
conv2_bias = tf.Variable(tf.truncated_normal([conv2_fnum],stddev=0.1,dtype=tf.float32))
conv2_output = tf.nn.conv2d(maxpool1,conv2_filter,
                            strides=[1,1,1,1],padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2_output,conv2_bias))
maxpool2 = tf.nn.max_pool(relu2, ksize=[1,maxpool2_ksize,maxpool2_ksize,1],
                          strides=[1,maxpool2_ksize,maxpool2_ksize,1], padding='SAME')

# conv3 + relu 
conv3_filter = tf.Variable(tf.truncated_normal([conv3_fsize,conv3_fsize,conv2_fnum,conv3_fnum],
                                               stddev=0.1,dtype=tf.float32))
conv3_bias = tf.Variable(tf.truncated_normal([conv3_fnum],stddev=0.1,dtype=tf.float32))
conv3_output = tf.nn.conv2d(maxpool2,conv3_filter,
                            strides=[1,1,1,1],padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3_output,conv3_bias))
maxpool3 = tf.nn.max_pool(relu3, ksize=[1,maxpool3_ksize,maxpool2_ksize,1],
                          strides=[1,maxpool3_ksize,maxpool3_ksize,1], padding='SAME')

conv_output = maxpool3

# In[]: fully connected layers
# conv3_output = tf.squeeze(conv3_output)
conv_shape = conv_output.get_shape().as_list()
conv_feature = conv_shape[1]*conv_shape[2]*conv_shape[3]
conv_output_flat = tf.reshape(conv_output,[conv_shape[0],conv_feature]) # shape must be specified

# fc1 + relu
fc1_w = tf.Variable(tf.truncated_normal([conv_feature,fc1],stddev=0.1,dtype=tf.float32))
fc1_b = tf.Variable(tf.truncated_normal([fc1],stddev=0.1,dtype=tf.float32))
fc1_output = tf.add(tf.matmul(conv_output_flat,fc1_w),fc1_b)
fc1_relu = tf.nn.relu(fc1_output)

# fc2 + 
fc2_w = tf.Variable(tf.truncated_normal([fc1,fc2],stddev=0.1,dtype=tf.float32))
fc2_b = tf.Variable(tf.truncated_normal([fc2],stddev=0.1,dtype=tf.float32))
fc2_output = tf.add(tf.matmul(fc1_relu,fc2_w),fc2_b)

fc_output = fc2_output
# In[]: construction of loss, training accuracy rate, testing accuracy rate

# loss_sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_pcl,logits=fc_output)
# opt_sig = tf.train.GradientDescentOptimizer(l_rate).minimize(loss_sig)
# fc_sig = tf.nn.sigmoid(fc_output) # network output

loss_soft = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_output,labels=batch_pcl)) # should be scalar
# opt_soft = tf.train.GradientDescentOptimizer(l_rate).minimize(loss_soft)
opt_soft = tf.train.MomentumOptimizer(l_rate, 0.9).minimize(loss_soft) # momentum optimizer
pred_soft = tf.nn.softmax(fc_output) # network output

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# In[]
def plate_accuracy(pcl,pred):
    """
    calculate the plate recognition accuracy
    pcl, true labels
    pred, predicted labels
    for the L_p characters in a plate, if the system only makes no greater than one wrong classification, we regard it as successful recognition
    """
    
    Pright_num = 0
    for ite_img in np.arange(batch_size):
        plate_start = ite_img*L_p
        plate_end = (ite_img+1)*L_p
        plate_sample_pred = pred[plate_start:plate_end]
        plate_sample_lab = pcl[plate_start:plate_end]
        if np.sum(plate_sample_pred!=plate_sample_lab) <= 1:
            Pright_num = Pright_num+1
    
    accP = Pright_num / float(batch_size)
    return accP

# In[]: evaluation
N_batch = Itrain_N // batch_size
loss_soft_train_arr = []
accI_train_arr = [] # plate recognition accuracy
accC_train_arr = [] # character recognition accuracy

loss_soft_test_arr = []
accI_test_arr = []
accC_test_arr = []

for ite_epoch in np.arange(N_epoch):
    # shuffled_train_PCI, shuffled_train_PCL = shuffle(train_PCI,train_PCL)
    
    rand_order = np.random.permutation(Itrain_N)
    
    Ibatch_ind_test = random.sample(range(Itest_N),batch_size)
    Cbatch_ind_test = pc_index(Ibatch_ind_test)
    val_pci_test = test_PCI[Cbatch_ind_test]
    val_pcl_test = test_PCL_1d[Cbatch_ind_test] # (batch_size*L_p,)
    val_dict_test = {batch_pci: val_pci_test, batch_pcl: val_pcl_test}
    
    for ite_batch in np.arange(N_batch):
        Ibatch_start = ite_batch*batch_size
        Ibatch_end = (ite_batch+1)*batch_size
        Ibatch_ind = rand_order[Ibatch_start:Ibatch_end]
        Cbatch_ind = pc_index(Ibatch_ind) # index of characters
    
        val_pci = train_PCI[Cbatch_ind]
        val_pcl = train_PCL_1d[Cbatch_ind]
        val_dict_train = {batch_pci: val_pci, batch_pcl: val_pcl}
    
        sess.run([opt_soft],feed_dict=val_dict_train) # update system before evaluation
        
        # training loss, training character recognition accuracy, training plate recognition accuracy evaluation
        loss_soft_train, pred_soft_train = sess.run([loss_soft, pred_soft],feed_dict=val_dict_train)
        loss_soft_train_arr.append(loss_soft_train)
        pred_soft_train = np.argmax(pred_soft_train,axis=1)
        accC_train = np.sum(pred_soft_train==val_pcl) / float(batch_size*L_p)
        accC_train_arr.append(accC_train)
        accI_train = plate_accuracy(val_pcl, pred_soft_train)
        accI_train_arr.append(accI_train)
        
        # testing loss, testing character recoginition accuracy, testing plate recognition accuracy evaluation
        loss_soft_test, pred_soft_test = sess.run([loss_soft, pred_soft], feed_dict=val_dict_test)
        loss_soft_test_arr.append(loss_soft_test)
        pred_soft_test = np.argmax(pred_soft_test,axis=1)
        accC_test = np.sum(pred_soft_test==val_pcl_test) / float(batch_size*L_p)
        accC_test_arr.append(accC_test)
        accI_test = plate_accuracy(val_pcl_test, pred_soft_test)
        accI_test_arr.append(accI_test)
        
        print('epoch {}/{},batch {}/{}: train loss (test loss), {} ({})'.format(ite_epoch, N_epoch, ite_batch, N_batch,
                                                                                loss_soft_train, loss_soft_test))
        print('epoch {}/{},batch {}/{}: train C-recog acc (test C-recog acc), {} ({})'.format(ite_epoch, N_epoch, ite_batch, N_batch,
                                                                                accC_train, accC_test))
        print('epoch {}/{},batch {}/{}: train P-recog acc (test P-recog acc), {} ({})'.format(ite_epoch, N_epoch, ite_batch, N_batch,
                                                                                accI_train, accI_test))
        

# In[]: training error, test accuracy visualization
fig_h = 10
fig_w = 10
plt.figure(figsize=(fig_h,fig_w))
plt.plot(loss_soft_train_arr,'-*',label='train loss')
plt.plot(loss_soft_test_arr,'-o',label='test loss')
plt.legend(loc='upper right')
plt.title('loss VS generations')
plt.xlabel('generation')
plt.ylabel('loss')
plt.show()

plt.figure(figsize=(fig_h,fig_w))
plt.plot(accC_train_arr,'-*', label='train C-recog acc')
plt.plot(accC_test_arr,'-o', label='test C-recog acc')
plt.title('C-recog acc VS generations')
plt.legend(loc='upper right')
plt.xlabel('generation')
plt.ylabel('C-recog accuracy')
plt.show()

plt.figure(figsize=(fig_h,fig_w))
plt.plot(accI_train_arr,'-*', label='train P-recog acc')
plt.plot(accI_test_arr,'-o', label='test P-recog acc')
plt.legend(loc='upper right')
plt.title('Precog acc VS generations')
plt.xlabel('generation')
plt.ylabel('P-recog accuracy')

# In[]: test accuracy, single character recognition accuracy, plate recognition 




# In[]: hidden layer feature visualization

def feature_vis():
    a = 0
    return a
