P607_50_Plates_Images.mat: Plates_Images_2.mat
P607_50_Plates_Images.mat: Plates_Images_3.mat
P607_50_Plates_Images.mat: Plates_Images_4.mat
P1010_52_Plates_Images.mat: Plates_Images_5.mat

0-9: position 1-10
A-Z: position 11-36

Each small individual character will be treated as independent sample for training, and in the
evaluation process, we will treat 7 as one

# @author: qi-qi based on jryi
# @contributor: jryi, Ke Ma
"""

import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import random
import pickle

import os


# In[]: data import and exploration
print('Data importing...')
N_class = 36  # choice of characters
C_h = 40  # character image height
C_w = 20  # character image width
I_h = 320  # car image height
I_w = 240  # car image width
L_p = 7  # plate length
N_parts = 9
N_sample = 50+60+50+52+29+60+32+92+51
batch_size = 80  # number of plates
MaxIte = 1000
fc2 = 100
fc3 = N_class

data_PCL = [0] * N_parts  # label for each character
data_PL = [0] * N_parts  # label for each plate
data_PCI = [0] * N_parts  # feature data for each character
data_PI = [0] * N_parts  # feature for each car image

for i in np.arange(N_parts):
    pcl = sio.loadmat('Plate_Character_Labels_%s.mat' % str(i))
    pcl = pcl["data_lab"]
    data_PCL[i] = pcl
    print (data_PCL[i].shape)

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



Itrain_N = int(0.8 * N_sample)
Itrain_ind = random.sample(range(N_sample), Itrain_N)  # sample from 0 to 231
Itest_N = N_sample - Itrain_N
Itest_ind = np.setdiff1d(np.arange(N_sample), Itrain_ind)

train_PL = data_PL[Itrain_ind]
train_PI = data_PI[Itrain_ind]
test_PL = data_PL[Itest_ind]
test_PI = data_PI[Itest_ind]

Ctrain_N = Itrain_N * L_p
Ctest_N = Itest_N * L_p

Ctrain_ind = []
for i in Itrain_ind:
    for j in np.arange(L_p):
        ind = i * L_p + j
        Ctrain_ind.append(ind)

Ctest_ind = np.setdiff1d(np.arange(N_sample * L_p), Ctrain_ind)

train_PCL = data_PCL[Ctrain_ind]
test_PCL = data_PCL[Ctest_ind]
train_PCL_1d = np.argmax(train_PCL,axis=1)
test_PCL_1d = np.argmax(test_PCL,axis=1)

train_PCI = data_PCI[Ctrain_ind] / 255.0  # scale down
train_PCI = np.reshape(train_PCI, (Ctrain_N, C_w, C_h))  # Data Transforamtion.
train_PCI = np.transpose(train_PCI, (0, 2, 1))
test_PCI = data_PCI[Ctest_ind] / 255.0
test_PCI = np.reshape(test_PCI, (Ctest_N, C_w, C_h))
test_PCI = np.transpose(test_PCI, (0, 2, 1))

# In[]: neural network hyper-parameters



# In[]: neural network construction, convolutional layers
# VALID: without padding; SAME: with padding
# batch_pcl = tf.placeholder(shape=(None,N_class),dtype=tf.float32)
# batch_pci = tf.placeholder(shape=(None,C_h,C_w),dtype=tf.float32)
vfsize = 3
vfnum = 16
training = True
data_format = 'NHWC' # channel last
def _residual_v1(x, vifnum):
    '''
    This is a deeper residual function of residual network.
    The implemention is according to Figure 5 in https://arxiv.org/pdf/1512.03385.pdf.
    :param x: Input Tensor.
    :return: Output Tensor.
    '''


    filter1 = tf.Variable(tf.truncated_normal([vfsize,vfsize, vifnum, vfnum], stddev= 0.1, dtype=tf.float32)) #define filter
    conv1_bias = tf.Variable(tf.truncated_normal([vfnum],stddev= 0.1, dtype=tf.float32))
    conv1_output = tf.nn.conv2d(x, filter1, strides=[1, 1, 1, 1], padding = 'SAME')
    conv1_output = tf.nn.bias_add(conv1_output, conv1_bias)
    conv1_output = batch_norm(conv1_output, training, data_format)


    conv2_input = tf.nn.relu(conv1_output)
    filter2 = tf.Variable(
        tf.truncated_normal([vfsize, vfsize, vfnum, vfnum], stddev=0.1, dtype=tf.float32))  # define filter
    conv2_bias = tf.Variable(tf.truncated_normal([vfnum], stddev=0.1, dtype=tf.float32))
    conv2_output = tf.nn.conv2d(conv2_input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_output = tf.nn.bias_add(conv2_output, conv2_bias)
    v_output = tf.add(conv2_output, x)
    v_output = batch_norm(v_output, training, data_format)

    v_output = tf.nn.relu(v_output)
    return v_output

def batch_norm(inputs, training, data_format):
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-5
    return tf.layers.batch_normalization(inputs=inputs, axis= 3 if data_format == 'NHWC' else 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


    # if self._data_format == 'channels_first':
    #   data_format = 'NCHW'
    # else:
    #   data_format = 'NHWC'


batch_pcl = tf.placeholder(shape=(batch_size*L_p,N_class),dtype=tf.float32)
batch_pci = tf.placeholder(shape=(batch_size*L_p,C_h,C_w),dtype=tf.float32)
batch_pci_ext = tf.expand_dims(batch_pci,axis=3) # 4d array for convolution operation,NHWC
print(batch_pci_ext.shape)
v_output = _residual_v1(batch_pci_ext, 1)
#v_output = _residual_v1(v_output, vfnum)
#v_output = _residual_v1(v_output, vfnum)
#v_output = _residual_v1(v_output, vfnum)
#v_output = _residual_v1(v_output, vfnum)
f_input = tf.nn.avg_pool(v_output,ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
[d1, d2, d3,d4] = f_input.get_shape().as_list()
fc1 = d2*d3*d4
print("fc1 num", fc1)
f_input = tf.reshape(f_input, (-1, fc1))
weights1 = tf.Variable(tf.truncated_normal([fc1, fc2], stddev=0.1, dtype=tf.float32)) ##1000
bias1 = tf.Variable(tf.truncated_normal([fc2], stddev=0.1, dtype=tf.float32))
f_output = tf.add(tf.matmul(f_input, weights1), bias1)
#f_input2= tf.nn.relu(f_output)
f_input2 = tf.nn.softmax(f_output)

weights2 = tf.Variable(tf.truncated_normal([fc2, fc3], stddev=0.1, dtype=tf.float32)) ##1000
bias2 = tf.Variable(tf.truncated_normal([fc3], stddev=0.1, dtype=tf.float32))
f_output = tf.add(tf.matmul(f_input2, weights2), bias2)
print (f_output.shape)
train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_pcl, logits=f_output))
print(tf.nn.softmax_cross_entropy_with_logits(labels=batch_pcl, logits=f_output))
l_rate = 0.001
optimizer = tf.train.AdamOptimizer(l_rate).minimize(train_loss)

f_output_soft = tf.nn.softmax(f_output)
correct_prediction = tf.equal(tf.argmax(f_output_soft, 1), tf.argmax(batch_pcl, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =0.5

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
# In[]

with tf.Session(config=config) as sess:
    sess.run(init)
    en_loss_train_arr = []
    en_loss_test_arr = []
    en_trainacc = []
    en_testacc = []
    en_trainPacc = []
    en_testPacc = []
    for k in np.arange(MaxIte):
        Img_ind = random.sample(range(Itrain_N),batch_size)
        test_Img_ind = random.sample(range(Itest_N),batch_size)
        Cbatch_ind = []
        test_Cbatch_ind = []
        for i in Img_ind:
            for j in np.arange(L_p):
                ind = i*L_p + j
                Cbatch_ind.append(ind)

        for i in test_Img_ind:
            for j in np.arange(L_p):
                ind = i*L_p + j
                test_Cbatch_ind.append(ind)

        val_pci = train_PCI[Cbatch_ind]
        val_pcl = train_PCL[Cbatch_ind]
        test_pci = test_PCI[test_Cbatch_ind]
        test_pcl = test_PCL[test_Cbatch_ind]
        #print(val_pci.shape)
        #print(val_pcl.shape)

        _, _en_loss_train, trainacc = sess.run([optimizer, train_loss, accuracy],
                                               feed_dict={batch_pci:val_pci,batch_pcl:val_pcl}) # matrix
        pred_train = sess.run([f_output_soft],feed_dict={batch_pci:val_pci,batch_pcl:val_pcl})
        pred_train = np.argmax(pred_train,axis=1)
        lab_train = np.argmax(val_pcl,axis=1)
        p_acc_train = plate_accuracy(lab_train,pred_train)
        en_trainPacc.append(p_acc_train)
        
        
        training = False
        _en_loss_test, testacc = sess.run([train_loss, accuracy],
                                             feed_dict={batch_pci:test_pci,batch_pcl:test_pcl}) # matrix
        pred_test = sess.run([f_output_soft],feed_dict={batch_pci:test_pci,batch_pcl:test_pcl})
        pred_test = np.argmax(pred_test,axis=1)
        lab_test = np.argmax(test_pcl,axis=1)
        p_acc_test = plate_accuracy(lab_test,pred_test)
        en_testPacc.append(p_acc_test)

        if k % 1 == 0:
            print ("Runing ...", k, "train/test loss ", _en_loss_train,"/", _en_loss_test, "trainacc", trainacc, "testacc", testacc)
            en_loss_train_arr.append(_en_loss_train)
            en_loss_test_arr.append(_en_loss_test)
            en_trainacc.append(trainacc)
            en_testacc.append(testacc)

    pickle.dump(en_loss_train_arr, open("train_loss", "wb"))
    pickle.dump(en_loss_test_arr, open("test_loss", "wb"))
    pickle.dump(en_trainacc, open("trainacc", "wb"))
    pickle.dump(en_testacc, open("testacc", "wb"))

plt.figure()
plt.plot(en_loss_train_arr,'-*',label='train loss')
plt.plot(en_loss_test_arr,'-o',label='test loss')
plt.legend(loc='upper right')
plt.show()


plt.figure()
plt.plot(en_trainacc,'-*',label='C-recog train accuracy')
plt.plot(en_testacc,'-o', label ='C-recog test accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(en_trainPacc,'-*',label='P-recog train accuracy')
plt.plot(en_testPacc,'-o',label='P-recog test accuracy')
plt.legend(loc='lower right')
plt.show()



# In[]: loss, error rate, hidden layer feature visualization
# bmean1 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
# bvar1 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
# bbeta1 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
# bgamm1 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
#conv1_output = tf.nn.batch_normalization(conv1_output, bmean1, bvar1, bbeta1, bgamm1, data_format = 'NHWC')# channel_last

#bmean2 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
#bvar2 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
#bbeta2 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
#bgamm2 = tf.Variable(tf.random_normal([vfnum], stddev=0.1, dtype=tf.float32), trainable=True)
#v_output = tf.nn.batch_normalization(v_output, bmean2, bvar2,  bbeta2, bgamm2, data_format = 'NHWC') # channel_last

