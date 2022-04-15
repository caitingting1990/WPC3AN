import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import scipy.io as scio
import numpy as np
data_filepath="Bearing/"
A_sets={'data':[],'label':[]}
B_sets={'data':[],'label':[]}
C_sets={'data':[],'label':[]}
D_sets={'data':[],'label':[]}
for filenames in os.listdir(data_filepath):
    subPath=os.path.join(data_filepath,filenames)
    for mat_filename in os.listdir(subPath):
        if mat_filename.endswith('.mat'):
            belong_set,belong_class=mat_filename.split('_')
            belong_class=int(belong_class.split(".")[0])
            mat_filepath=os.path.join(subPath,mat_filename)
            tmp_data=scio.loadmat(mat_filepath)
            for key in tmp_data.keys():
                if key.endswith('DE_time'):
                    if belong_set=='A':
                        A_sets['data'].append(tmp_data[key][:,0])
                        A_sets['label'].append(belong_class)
                    if belong_set=='B':
                        B_sets['data'].append(tmp_data[key][:,0])
                        B_sets['label'].append(belong_class)
                    if belong_set=='C':
                        C_sets['data'].append(tmp_data[key][:,0])
                        C_sets['label'].append(belong_class)
                    if belong_set=='D':
                        D_sets['data'].append(tmp_data[key][:,0])
                        D_sets['label'].append(belong_class)

def seq_chunk(seq,chunk,step=None):
    if step==None:
        step=chunk
    seq_length=seq.shape[0]
    rows=(seq_length-chunk)//step+1
    new_set=np.empty(shape=(rows,chunk))
    for i in range(0,seq_length-chunk+1,step):
        new_set[i//step]=seq[i:i+chunk]
    return new_set

def shuffle_sets(data,label):
    np.random.seed(1000)
    permutation = np.random.permutation(data.shape[0])
    data_shuffled = data[permutation,]
    label_shuffled = label[permutation,]
    return data_shuffled,label_shuffled

def get_xy(A_sets,arr_len=1024,step=300,x_shape=(-1,32,32,1)):
    A_x=[]
    A_y=[]
    for data,label in zip(A_sets['data'],A_sets['label']):
        tmp_data=np.reshape(seq_chunk(data,arr_len,step=step),newshape=x_shape)
        A_x.append(tmp_data)
        A_y.append((np.ones(shape=(tmp_data.shape[0]))*label).astype(int))
    A_x=np.array(A_x)
    A_y=np.array(A_y)
    A_x=np.concatenate(A_x,axis=0)
    A_y=np.concatenate(A_y,axis=0)
    return shuffle_sets(A_x,A_y)

All_sets=(A_sets,B_sets,C_sets,D_sets)
set_name=(("A_data_1024","A_label_1024"),("B_data_1024","B_label_1024"),("C_data_1024","C_label_1024"),("D_data_1024","D_label_1024"))
for step,tmp_set in enumerate(All_sets):
    x_s,y_s=get_xy(D_sets,arr_len=1024,step=100,x_shape=(-1,1024))
    x_s=x_s[:12000]
    y_s=y_s[:12000]
    np.save("oral_data/"+set_name[step][0]+".npy",x_s)
    np.save("oral_data/"+set_name[step][1]+".npy",y_s)
