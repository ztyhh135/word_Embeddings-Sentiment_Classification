import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string
import collections
import inspect


batch_size = 50
lstm_size = 128
state_size = 5
number_of_layers=3
lr = 0.0005

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    
    filename = "reviews.tar.gz"
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath('__file__')),'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(os.path.realpath('__file__'))
            tarball.extractall(os.path.join(dir, 'data2/'))
    data = []
    dir = os.path.dirname(os.path.realpath('__file__'))
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    print("Parsing %s files" % len(file_list))
    for f in file_list:
        with open(f, "r",encoding = 'utf-8') as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            words  = no_punct.split()
            if len(words)<40:
                pad = ['UNK' for i in range(40-len(words))]
                words.extend(pad)
            elif len(words)>=40:
                words = words[:40]
            row = []
            for word in words:
                #print(word)
                #print(word in glove_dict)
                if word in glove_dict:
                    row.append(glove_dict[word])
                else:
                    row.append(glove_dict['UNK'])
            data.append(row)
    data = np.array(data)
    
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    word_index_dict = {}
    word_index_dict['UNK'] = 0
    word_index = 1
    word_vec = [[0]*50]
    for i in data:
        word_vec_txt = i.split(' ')
        word  = word_vec_txt[0]
        word_vec.append(word_vec_txt[1:])
        word_index_dict[word] = word_index
        word_index+=1
    embeddings = np.array(word_vec,dtype='float32')
    return embeddings, word_index_dict

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    
    ###input
#    with tf.device('/cpu:0'):
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape=())
    print(dropout_keep_prob)
    input_data = tf.placeholder(tf.float64, shape=[batch_size,40],name="input_data")
    input_data_trans = tf.cast(input_data,tf.int32)
    #input_data.dtype = 'int32'
    labels = tf.placeholder(tf.int32, shape=[batch_size,2],name="labels")
    labels_ = tf.to_float(labels)
    embeddings = tf.nn.embedding_lookup(glove_embeddings_arr,input_data_trans)###shape(50,80,50)
    ##bulid lstm
    ##lstm_size is the number of units in hidden layers.
   
    def lstm_cell():
        if 'reuse' in inspect.signature(tf.contrib.rnn.BasicLSTMCell.__init__).parameters:
            return tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0,
                                     state_is_tuple=True, 
                                     reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                                     lstm_size, forget_bias=1.0, state_is_tuple=True)
    
    def drop_cell():
        return  tf.contrib.rnn.DropoutWrapper(lstm_cell(), input_keep_prob= dropout_keep_prob, output_keep_prob= dropout_keep_prob)
    

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(number_of_layers)], state_is_tuple=True)
    initial_state  = stacked_lstm.zero_state(batch_size, tf.float32)
    state = initial_state
    outputs = []
    with tf.variable_scope("RNN"):
        for i in range(40):
    # The value of state is updated after processing each batch of words.
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            cell_out, state = stacked_lstm(embeddings[:, i], state)
        ###cell_out.shape->TensorShape([Dimension(50), Dimension(20)])
            outputs.append(cell_out)
    ###len(outputs)=40
    final_state = state
    print("after lstm")
    #print(outputs)
    output = outputs[-1]
    W = tf.Variable(tf.truncated_normal([lstm_size, 2], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1,shape=[2]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(output, W) + bias)
    #loss = -tf.reduce_mean(tf.reduce_mean(labels * tf.log(y_pre) ))

    loss = -tf.reduce_mean(labels_ * tf.log(y_pre))
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
