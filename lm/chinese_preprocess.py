import tensorflow as tf
import jieba
import numpy as np
def get_chinese_indexlist(text):
    seg_list=jieba.cut(text,cut_all=True)
    word_index={}
    index_word={}
    temp=1
    for i in seg_list:
        if word_index.get(i,"NONE")!="NONE":
            continue
        word_index[i]=temp
        index_word[temp]=i
        temp=temp+1
    return word_index,index_word

def text_to_index(text,word_index):#single string change
    seg_list=jieba.cut(text,cut_all=True)
    temp=[]
    for i in seg_list:
        temp.append(word_index.get(i,0))
    return temp

def pad_sequence(sequence):#pad list
    lens=[len(seq) for seq in sequence]
    maxlen=max(lens)
    list=[]
    pad_seq=(seq+[0]*(maxlen-len(seq)) for seq in sequence)
    for i in pad_seq:
        list.append(i)
    return list

def read_file(path_name):
    with open(path_name,'r',encoding="utf8") as myfile:
        contents=myfile.read(1000)
        return contents

word = read_file("wiki_zh_2019.txt")
word1 =['十五的月亮，月亮的十五','见到你很高兴']

word_index,index_word=get_chinese_indexlist(word)
print(word_index)
tensor=[]
for change in word1:
    change=text_to_index(change,word_index)
    tensor.append(change)
tensor=pad_sequence(tensor)
text=np.array(tensor)
print(text)
word=tf.data.Dataset.from_tensor_slices(text)
def split_input_target(chunk):
    input_text=chunk[:-1]
    target_text=chunk[1:]
    return input_text,target_text
dataset=word.map(split_input_target)
print(dataset)
BATCH_SIZE=16
BUFFER_SIZE=10000