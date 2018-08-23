"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
from sklearn.cross_validation import train_test_split
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 100000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 11499)
            labels.append([1, 0])
        else:
            num = randint(11500, 22999)
            labels.append([0, 1])
        arr[i] = train[num]
    return arr, labels
def split(train):
    train_2,test_2 = train_test_split(train[12500:],test_size=0.08)
    train_1,test_1 = train_test_split(train[:12500],test_size=0.08)
    train_1 = list(train_1)
    train_2 = list(train_2)
    np.random.shuffle(test_1)
    np.random.shuffle(test_2)
    test_1 = list(test_1)
    test_2 = list(test_2)
    train_1.extend(train_2)
    test_1.extend(test_2)
    train_s = np.array(train_1)
    test_s = np.array(test_1)
    return train_s,test_s
def get_test_y():
    labels1 = []
    labels2 = []
    for i in range(1000):
        labels1.append([1, 0])
        labels2.append([0, 1])
    labelright = labels1
    labelright.extend(labels2)
    labelwrong = labels2
    labelwrong.extend(labels1)
    LR=[]
    LW=[]
    for i in range(30):
        LR.append(labelright[batch_size*i:batch_size*i+batch_size])
        LW.append(labelwrong[batch_size*i:batch_size*i+batch_size])
    return LR,LW
def get_test(test):
    t = list(test)
    T = []
    for i in range(30):
        T.append(t[i*batch_size:i*batch_size+batch_size])
    tt=np.array(T)
    return tt
    
# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
#########split 0.08:
train,t = split(training_data)
test = get_test(t)
righty,wrongy = get_test_y()

input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)

summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
acc = []
for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob:0.75})
  
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        writer.add_summary(summary, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
    if (i % 5000 == 0 and i >= 30000 and i<=50000):
        lracc = 0
        
        for j in range(6):
            s = (i-30000)//5000
            loss_value, accuracy_value = sess.run(
                    [loss, accuracy],
                    {input_data: test[j+s*6],
                     labels: righty[j+s*6]})
            ##writer.add_summary(summary, j)
            lracc +=accuracy_value
            ##writer.add_summary(summary, j)
        lracc = lracc/6
        acc.append(lracc)
        print("LRacc", lracc)
        
        
    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)
sess.close()
