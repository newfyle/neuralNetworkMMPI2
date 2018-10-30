#!/usr/bin/env python
# coding: utf-8

# In[323]:


import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz
import graphviz
import ast
import re
import unittest


# In[312]:


batch_size = 800
num_of_classes = 11
epochs = 4000


# In[328]:


def return_list_random():
    list=[]
    with open("random_scales.txt","r") as f:
        for line in f:
            line_mod=line.strip("\n")
            list.append(ast.literal_eval(line_mod))
    return(list)

raw_list=return_list_random()
#data must be pre processed into desired categories from it's description


# In[329]:


def test_list():
    assert(len(return_list_random())==800)
test_list()


# In[314]:


def split_value(list,training,test):
    one_train=[]
    two_train=[]
    one_test=[]
    two_test=[]
    total_classes=[]
    if(training+test!=len(list)):
        print("Not same ratio")
        throw(Exception)
    else:
        try:
            for a in range(0,training):
                value=list[a][0]
                one_train.append(value)
                two_train.append(match_words(list[a][1]))
            for b in range(training,training+test):
                value=list[a][0]
                one_test.append(value)
                two_test.append(match_words(list[a][1]))
        except:
            print("Empty List")
    return(list_to_np([one_train,one_test,two_train,two_test,total_classes]))
# this is used to split value from it's possible categorization


# In[345]:


def test_splitting():
    list_prototype=[[[1,2,3],["Mild Psychasthenia"]],[[2,2,3],["Mild Schizoprenia"]],[[2,2,3],["Mild Schizoprenia"]]]
    a=split_value(list_prototype,2,1)
    assert(a[0].shape[0]==2)
    assert(a[1].shape[0]==1)
test_splitting()


# In[336]:


#remove
def match_words(sentence):
    list=['No interpretation of', 'Mild', 'Moderate', 'Severe', 'Reverse severe', 'Traditional', 'Tendency of', 'Rejecting']
    for a in list:
        if(re.compile(r'\b({0})\b'.format(a), flags=re.IGNORECASE).search(sentence[0])!=None):
            split=sentence[0].split(a)[1].strip(" ")
            b=['Hypomania', 'Social Introversion', 'Psychasthenia', 'Feminity', 'Schizoprenia', 'Hypochondriasis', 'Masculinity', 'Paranoia', 'Depression', 'Hysteria', 'Psychopathic Deviate']
            try:
                index=b.index(split)
                return(index)
            except:
                continue


# In[337]:


def list_to_np(list):
    list_return=[]
    for a in list:
        list_return.append(np.array(a))
    return(list_return)


# In[348]:


def train(list_specified,ratio_big,ratio_small,shape_input,first_hidden_layer_no,end_class):
    x_train,x_test,y_train,y_test,total_classes=split_value(list_specified,ratio_big,ratio_small) #ratio of training to test values
    X_train=x_train.reshape(x_train.shape[0],1,shape_input,1) #shape, translates to depth of 1 with 36 elements in one array
    X_test=x_test.reshape(x_test.shape[0],1,shape_input,1)
    
    x_train = x_train.astype('float32') #change value to float
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes) #convert the categories of the y obtained from the data into numeric classes
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model=Sequential()
    model.add(Dense(first_hidden_layer_no, activation='relu', input_shape=(11,))) #512 is the specified hidden layer number, and input node is the input shape 11 inputs
    model.add(Dropout(0.2))
    model.add(Dense(end_class, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    ann_viz(model, title="Neural Network System")
train(raw_list,600,200,11,512,num_of_classes)

