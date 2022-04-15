from PyEMD import EEMD
import tensforflow as tf
from tensorflow import keras
from oral_data import *

def kurt(arrs):
    '''
    计算峭度[n,1024]
    :return: [n,]
    '''
    m=np.mean(arrs,axis=1)#获取均值
    m=np.expand_dims(m,axis=-1)
    upper=np.mean((arrs-m)**4,axis=1)
    down=(np.sum((arrs-m)**2,axis=1)/arrs.shape[1])**2
    return upper/down

def eemd_restruct(signals,k=5):
    '''
    基于emmd重构信号
    :param signals:输入信号
    :param k: 选择重构的最大峭度对应的n个IMF函数组
    :return: 重构信号
    '''
    if len(signals.shape)==1:
        eemd=EEMD()
        eimfs=eemd(signals)
        mask=np.argsort(kurt(eimfs))[-k:]
        features=eimfs[mask]
        return np.sum(features,axis=0)
    else:
        out_signal=[]
        for signal in signals:
            out_signal.append(eemd_restruct(signals=signal,k=k))
        return np.array(out_signal)

class DAE(keras.Model):
    def __init__(self):
        super(DAE, self).__init__()
        self.dae_input=keras.Input(shape=(1024,))
        self.input_layer=keras.layers.Dense(1024,activation='sigmoid')
        self.dp=keras.layers.Dropout(0.2)
        self.dae_hidden=keras.layers.Dense(1024,activation='sigmoid')
        self.dae_out=keras.layers.Dense(1024)
        self.call(self.dae_input)
    def call(self,inputs,training=None):
        tmp=self.input_layer(inputs)
        tmp=self.dp(tmp)
        tmp1=self.dae_hidden(tmp)
        tmp=self.dae_out(tmp1)
        return tmp,tmp1


x_s,y_s=get_xy(A_sets,arr_len=1024,step=1024,x_shape=(-1,1024))
x_s_eemd=eemd_restruct(x_s[:1200])
x_t, y_t = get_xy(B_sets, arr_len=1024, step=1024, x_shape=(-1, 1024))
x_t_eemd =eemd_restruct(x_t[:1200])
total_x=np.concatenate((x_s_eemd,x_t_eemd),axis=0)

dae=DAE()
dae.compile(optimizer='adam',loss=['mse'])
dae.fit(x= total_x,y= total_x,batch_size=64,epochs=200,verbose=0)
# 最终获得的属于预处理结果,取隐层数据
out_s,x_s_hidden=dae.predict(x_s_eemd)
out_t,x_t_hidden=dae.predict(x_t_eemd)

# 所有处理后的数据文件存储为npy文件A_data.npy
np.save("oral_data/A_data.npy",x_s_hidden)
np.save("oral_data/A_label.npy",y_s)
np.save("oral_data/B_data.npy",x_t_hidden)
np.save("oral_data/B_label.npy",y_t)
