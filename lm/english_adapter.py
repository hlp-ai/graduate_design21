import io
import re
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import  train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def splitsentence(file_path):
    input = []
    csv_reader = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    for row in csv_reader:
        row = re.sub(r"=[\s\S]*=", "", row)
        row = re.sub(r"[^a-zA-Z?.!,¿]+", " ", row)
        row = re.sub(r"[ ]+", " ", row)
        row=re.split(r"[?.!¿]",row)
        for line in row:
            if line != " " and line != "":
                print(line)
                input.append(line)
    return input
def tokenize(input):
    tokenizer=Tokenizer(filters='')
    tokenizer.fit_on_texts(input)
    tensor=tokenizer.texts_to_sequences(input)
    tensor=pad_sequences(tensor,padding='post')
    return tensor,tokenizer
def getdataset(batch_size,buffer_size,pathname):

    test=splitsentence(pathname)
    tensor,tokenizer=tokenize(test)
    text_train,text_valid=train_test_split(tensor,test_size=0.2)
    len1=len(text_train)
    len2=len(text_valid)
    word_train=tf.data.Dataset.from_tensor_slices(text_train)
    word_valid=tf.data.Dataset.from_tensor_slices(text_valid)
    def split_input_target(chunk):
        input_text=chunk[:-1]
        target_text=chunk[1:]
        return input_text,target_text
    dataset_train=word_train.map(split_input_target)
    dataset_valid=word_valid.map(split_input_target)
    dataset_train = dataset_train.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset_valid = dataset_valid.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset_train,dataset_valid,len1,len2,len(tokenizer.word_index)