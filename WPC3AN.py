#初始化插件包
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "GPU未接入使用"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import datetime
from tensorflow.keras.utils import Progbar
#初始化部分参数
batch_size=128
all_acc=[]

def get_cos_dist(X1,X2):
    X1_norm=tf.sqrt(tf.reduce_sum(tf.square(X1),axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    X1_X2=tf.matmul(X1,tf.transpose(X2))
    X1_X2_norm=tf.matmul(tf.reshape(X1_norm,[X1.shape[0],1]),tf.reshape(X2_norm,[1,X2.shape[0]]))
    cos=X1_X2/X1_X2_norm
    return cos

@tf.custom_gradient
def gradient_reversal(x,alpha=1.0):
	def grad(dy):
		return -dy * alpha, None
	return x, grad
class GRL(keras.layers.Layer):
    def __init__(self):
        super(GRL, self).__init__()
    def call(self, x,alpha=1.0):
        return gradient_reversal(x, alpha)

#所有关键模块
class ParallelConv(keras.layers.Layer):
    '''
    多尺度残差卷积网络模块
    channels 表示目标通道
    '''
    def __init__(self,channels,kernel_sizes=(16,7,5,3),strides=1):
        super(ParallelConv, self).__init__()
        self.cvs=[]
        for size in kernel_sizes:
            self.cvs.append(keras.layers.Conv2D(filters=channels,kernel_size=(size,size),strides=strides,padding='same'))
        self.cv1=keras.layers.Conv2D(filters=channels*len(kernel_sizes),kernel_size=(1,1),strides=strides,padding='same')
    def call(self, inputs, **kwargs):
        tmp=[]
        for cv in self.cvs:
            tmp.append(cv(inputs))
        inputs_change=self.cv1(inputs)
        tmp=tf.concat(tmp,axis=-1)
        return tf.keras.layers.add((inputs_change,tmp))
    def get_config(self):
        config=super(ParallelConv, self).get_config().copy()
        config.update({
            'parallel_cvs':self.cvs,
            'parallel_cv1':self.cv1
        })
        return config

class FCModule(tf.keras.layers.Layer):
    '''
    特征修正模块：Features Corrected Module
    channels 特征通道数量
    '''
    def __init__(self,channels,name="fc_module"):
        super(FCModule, self).__init__()
        self.fc0=tf.keras.layers.Conv2D(filters=channels,kernel_size=(1,1),strides=1,padding='same')
        self.fc1=tf.keras.layers.Conv2D(filters=channels,kernel_size=(1,1),strides=1,padding='same')
    def call(self, inputs, **kwargs):
        reshape=len(tf.shape(inputs))<4
        if reshape:
            inputs=tf.reshape(inputs,shape=(-1,1,1,tf.shape(inputs)[-1]))
        tmp=self.fc0(inputs)#[b,n,c]
        tmp=tf.nn.relu(tmp)
        tmp=self.fc1(tmp)#[b,n,c]
        tmp=tf.nn.relu(tmp)
        res=inputs+tmp
        if reshape:
            res=tf.reshape(res,shape=(-1,tf.shape(inputs)[-1]))
        return res
    def get_config(self):
        config=super(FCModule, self).get_config().copy()
        config.update({
            'fc_fc0':self.fc0,
            'fc_fc1':self.fc1
        })
        return config

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t= tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    n_samples =n_s+n_t
    total = tf.concat([source, target], axis=0)                                                      #   [None,n]
    total0 = tf.expand_dims(total,axis=0)               #   [1,b,n]
    total1 = tf.expand_dims(total,axis=1)               #   [b,1,n]
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),axis=2)     #   [b,b,n]=>[b,b]                                 #   [None,None,n]=>[128,128,1]
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / float(n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)   #[b,b]

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t= tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    XX = tf.reduce_sum(kernels[:n_s, :n_s])/float(n_s**2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/float(n_t**2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/float(n_s*n_t)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/float(n_s*n_t)
    loss = XX + YY - XY - YX
    return loss

def Entropy(y_true,y_pred):
    y_pred=tf.nn.softmax(y_pred)
    mask=y_pred>1e-7
    mask_out=y_pred[mask]
    loss=tf.reduce_mean(-mask_out*(tf.math.log(mask_out)/tf.math.log(10.0)))
    return loss

def feature_extractor(inputs):#(32,32,1)=>(256,)
    tmp=ParallelConv(16, kernel_sizes=(11, 8, 5, 3), strides=1)(inputs)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.MaxPooling2D((2, 2), 2, padding='same')(tmp)
    tmp0=keras.layers.Activation('relu')(keras.layers.Conv2D(64,(1,1),2,'same')(inputs))
    tmp1=keras.layers.add((tmp,tmp0))
    tmp = ParallelConv(32, kernel_sizes=(5, 3), strides=1)(tmp1)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Activation('relu')(tmp)
    tmp = keras.layers.MaxPooling2D((2, 2), 2, padding='same')(tmp)
    tmp0 = keras.layers.Activation('relu')(keras.layers.Conv2D(64, (1, 1), 2, 'same')(tmp1))
    tmp1 = keras.layers.add((tmp, tmp0))
    tmp = ParallelConv(64, kernel_sizes=(5, 3), strides=1)(tmp1)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Activation('relu')(tmp)
    tmp = keras.layers.MaxPooling2D((2, 2), 2, padding='same')(tmp)
    tmp0 = keras.layers.Activation('relu')(keras.layers.Conv2D(128, (1, 1), 2, 'same')(tmp1))
    tmp1 = keras.layers.add((tmp, tmp0))
    tmp = ParallelConv(128, kernel_sizes=(5, 3), strides=1)(tmp1)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Activation('relu')(tmp)
    tmp = keras.layers.MaxPooling2D((2, 2), 2, padding='same')(tmp)
    tmp0 = keras.layers.Activation('relu')(keras.layers.Conv2D(256, (1, 1), 2, 'same')(tmp1))
    tmp1 = keras.layers.add((tmp, tmp0))
    tmp = ParallelConv(128, kernel_sizes=(5, 3), strides=1)(tmp1)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Activation('relu')(tmp)
    tmp = keras.layers.GlobalAveragePooling2D()(tmp)
    return keras.Model(inputs=inputs,outputs=tmp)

def classifer():
    return keras.Sequential([
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64),
        keras.layers.Activation('relu'),
        keras.layers.Dense(10)
    ])
def domain_classifer():
    return keras.Sequential([
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16),
        keras.layers.Activation('relu'),
        keras.layers.Dense(2)
    ])

#%%
def learning_rate_schedule(process,init_learning_rate = 0.01,alpha = 10.0 , beta = 0.75):
    """
    这个学习率的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param init_learning_rate: 初始学习率，默认为0.01
    :param alpha: 参数alpha，默认为10
    :param beta: 参数beta，默认为0.75
    """
    return init_learning_rate /(1.0 + alpha * process)**beta

def grl_lambda_schedule(process,gamma=10.0):
    """
    这是GRL的参数lambda的变换函数
    :param process: 训练进程比率，值在0-1之间
    :param gamma: 参数gamma，默认为10
    """
    return 2.0 / (1.0+np.exp(-gamma*process)) - 1.0

class WDCNN_plus(object):
    def __init__(self):
        """
        这是MNINST与MNIST_M域适配网络的初始化函数
        :param config: 参数配置类
        """
        # 初始化参数类
        self.epoch =80
        self.metric_weights=[1,1,1,1,1,1,1]
        self.transfer_names=('A','B')
        # 定义相关占位符
        self.grl_lambd = 1.0              # GRL层参数
        self.init_learningrate=1e-2
        # 搭建深度域适配网络
        self.build_DANN()

        # 定义训练和验证损失与指标
        self.loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.acc=keras.metrics.SparseCategoricalAccuracy()
        self.train_loss =keras.metrics.Mean("train_loss", dtype=tf.float32)
        self.src_clf_loss = keras.metrics.Mean("src_clf_loss", dtype=tf.float32)
        self.src_dc_loss = keras.metrics.Mean("src_dc_loss", dtype=tf.float32)
        self.tar_dc_loss =keras.metrics.Mean("tar_dc_loss", dtype=tf.float32)
        self.mmd_loss=keras.metrics.Mean("mmd_loss", dtype=tf.float32)
        self.src_mmd_loss=keras.metrics.Mean("src_mmd_loss", dtype=tf.float32)
        self.src_clf_acc = keras.metrics.Mean("src_clf_acc", dtype=tf.float32)
        self.tar_clf_acc = keras.metrics.Mean("tar_clf_acc", dtype=tf.float32)
        self.src_dc_acc = keras.metrics.Mean("src_dc_acc", dtype=tf.float32)
        self.tar_dc_acc = keras.metrics.Mean("tar_dc_acc", dtype=tf.float32)
        self.val_loss =keras.metrics.Mean("val_loss", dtype=tf.float32)
        self.val_acc =keras.metrics.Mean("val_acc", dtype=tf.float32)
        # 定义优化器
        self.optimizer = tf.keras.optimizers.SGD(self.init_learningrate,
                                                 momentum=0.9)
        # 初始化早停策略
        self.early_stopping = False

    def build_DANN(self):
        """
        这是搭建域适配网络的函数
        :return:
        """
        # 定义源域、目标域的图像输入和DANN模型图像输入
        src_inputs = keras.Input(shape=(32, 32, 1))
        tar_inputs = keras.Input(shape=(32, 32, 1))
        self.feature_extractor = feature_extractor(src_inputs)
        self.clf_classifer = classifer()
        self.dc_classifer = domain_classifer()
        self.grl = GRL()
        self.fcm0 = FCModule(256)
        self.fcm1 = FCModule(10)
        src_fea=self.feature_extractor(src_inputs)
        tar_fea=self.fcm0(self.feature_extractor(tar_inputs))
        src_dc=self.dc_classifer(self.grl(src_fea))
        tar_dc=self.dc_classifer(self.grl(tar_fea))
        src_clf=self.clf_classifer(src_fea)
        tar_clf=self.clf_classifer(tar_fea)
        tar_clf=self.fcm1(tar_clf)
        self.dann_model = keras.Model(inputs=[src_inputs,tar_inputs],outputs=[src_clf,tar_clf,src_dc,tar_dc])
        self.dann_model.summary()
        # self.dann_model.summary()

    def train(self,s_sets,t_sets,train_iter_num,val_iter_num):
        """
        这是DANN的训练函数
        :param train_source_datagen: 源域训练数据集生成器
        :param train_target_datagen: 目标域训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param train_iter_num: 每个epoch的训练次数
        :param val_iter_num: 每次验证过程的验证次数
        """
        # 初始化相关文件目录路径
        # print('\n----------- start to train -----------\n')
        time = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        print(time)
        self.all_acc=[]
        self.stop_num=0
        # self.progbar = Progbar(self.epoch+1)
        for ep in np.arange(1,self.epoch+1,1):
            # 初始化精度条
            # self.progbar.update(ep)
            self.progbar1 = Progbar(train_iter_num+1)
            s_iter=iter(s_sets.batch(batch_size))
            t_iter=iter(t_sets.batch(batch_size))
            val_iter = iter(t_sets.batch(batch_size))
            # 进行一个周期的模型训练
            train_loss,src_clf_acc=self.train_one_epoch(s_iter,t_iter,train_iter_num,ep)
            # 进行一个周期的模型验证
            val_loss,val_acc=self.eval_one_epoch(val_iter,val_iter_num,ep)
            # 更新进度条
            self.progbar1.update(train_iter_num+1, [('val_loss', self.val_loss.result()),
                                                   ("val_acc", self.val_acc.result())])
            # 保存训练过程中的模型
            # str = "Epoch{:03d}-train_loss-{:.3f}-val_loss-{:.3f}-src_acc-{:.3f}-val_acc-{:.3f}" \
            #     .format(ep, self.train_loss.result(), self.val_loss.result(), self.src_clf_acc.result(), self.val_acc.result())
            # self.progbar.update(ep,[("train_loss",train_loss),
            #                         ("val_loss",val_loss),
            #                         ("src_clf_acc", src_clf_acc),
            #                         ("val_acc", val_acc)])
            self.all_acc.append(val_acc)
            print('\rEpoch {}/{} total_loss:{:.4f} mmd_loss:{:.4f} src_mmd_loss:{:.4f} src_clf_acc:{:.4f} val_acc:{:.4f} max_val_acc:{:.4f}'.format(ep,
                    self.epoch,self.train_loss.result(),
                    self.mmd_loss.result(),
                    self.src_mmd_loss.result(),
                    src_clf_acc,
                    val_acc,max(self.all_acc),
                  ),
                  end='',
                  flush=True)
            # print(str)
            # 损失和指标清零
            self.train_loss.reset_states()
            self.src_clf_loss.reset_states()
            self.src_dc_loss.reset_states()
            self.tar_dc_loss.reset_states()
            self.mmd_loss.reset_states()
            self.src_mmd_loss.reset_states()
            self.src_clf_acc.reset_states()
            self.tar_clf_acc.reset_states()
            self.src_dc_acc.reset_states()
            self.tar_dc_acc.reset_states()
            self.val_acc.reset_states()
            self.val_loss.reset_states()
            if self.all_acc[-1] > max(self.all_acc):
                self.dann_model.save(
                    "WPC3AN_WDCNN//wpc3an_{}{}.h5".format(self.transfer_names[0], self.transfer_names[1]))
            if len(self.all_acc)>=10:
                #连续5次下降
                if self.all_acc[-1]<=self.all_acc[-2] and self.all_acc[-2]<=self.all_acc[-3] and self.all_acc[-3]<=self.all_acc[-4]:
                    self.stop_num=self.stop_num+1
                    if self.stop_num>2:
                        self.early_stopping=True
            if ep==self.epoch or self.early_stopping:
                self.early_stopping=False
                tmp_acc = np.array(self.all_acc)
                tmp_acc = np.sort(tmp_acc )
                acc = tmp_acc[-10:]
                print("mean:{},hurt:{}".format(np.mean(acc),(np.max(acc) - np.mean(acc))))
                # self.progbar.update(ep+1,[("mean",np.mean(acc)),("hurt",(np.max(acc) - np.mean(acc)))])
                break
            time = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            print(time)
        # print('\n----------- end to train -----------\n')


    def train_one_epoch(self,s_iter,t_iter,train_iter_num,ep):
        '''
         这是一个周期模型训练的函数
        :param x_s:  源域训练数据集生成器
        :param y_s:  目标域训练数据集生成器
        :param x_t:  源域训练数据集生成器
        :param y_t:  目标域训练数据集生成器
        :param train_iter_num: 一个训练周期的迭代次数
        :param ep: 当前训练周期
        :return:
        '''
        for i in np.arange(1, train_iter_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            # 更新学习率并可视化
            x_s, y_s, d_s = s_iter.__next__()
            x_t, y_t, d_t = t_iter.__next__()
            iter = (ep - 1) * train_iter_num + i
            process = iter * 1.0 / (self.epoch* train_iter_num)
            self.grl_lambd = grl_lambda_schedule(process)
            learning_rate = learning_rate_schedule(process, init_learning_rate=self.init_learningrate)
            tf.keras.backend.set_value(self.optimizer.lr, learning_rate)
            # 计算图像分类损失梯度
            with tf.GradientTape() as tape:
                # 计算图像分类预测输出、损失和精度
                src_features = self.feature_extractor(x_s,training=True)
                tar_features = self.fcm0(self.feature_extractor(x_t,training=True))
                src_dc = self.dc_classifer(self.grl(src_features, self.grl_lambd),training=True)
                tar_dc = self.dc_classifer(self.grl(tar_features, self.grl_lambd),training=True)
                src_clf = self.clf_classifer(src_features,training=True)
                tar_clf = self.fcm1(self.clf_classifer(tar_features,training=True))
                loss1=self.loss(y_s,src_clf)
                loss2=self.loss(d_s,src_dc)
                loss3=self.loss(d_t,tar_dc)
                loss4=Entropy(None,tar_clf)
                loss5=MMD(src_features,tar_features)
                loss6=MMD(src_clf,tar_clf,fix_sigma=1)
                loss7=self.train_one_epoch_end(src_features,y_s)
                acc1=self.acc(y_s,src_clf)
                self.acc.reset_states()
                acc2=self.acc(y_t,tar_clf)
                self.acc.reset_states()
                acc3=self.acc(d_s,src_dc)
                self.acc.reset_states()
                acc4=self.acc(d_t,tar_dc)
                self.acc.reset_states()

                # 计算训练损失、图像分类精度和域分类精度
                loss = self.metric_weights[0]*tf.reduce_mean(loss1) + \
                       self.metric_weights[1]*tf.reduce_mean(loss2)+\
                       self.metric_weights[2]*tf.reduce_mean(loss3)+\
                       self.metric_weights[3]*tf.reduce_mean(loss4)+\
                       self.metric_weights[4]*loss5+\
                       self.metric_weights[5]*loss6+ \
                       self.metric_weights[6]*loss7
            # 自定义优化过程
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            # 计算平均损失与精度
            self.train_loss(loss)
            self.src_clf_loss(loss1)
            self.src_dc_loss(loss2)
            self.tar_dc_loss(loss3)
            self.mmd_loss(loss5+loss6)
            self.src_mmd_loss(loss7)
            self.src_clf_acc(acc1)
            self.tar_clf_acc(acc2)
            self.src_dc_acc(acc3)
            self.tar_dc_acc(acc4)

            # 更新进度条
            self.progbar1.update(i, [('loss', self.train_loss.result()),
                                     ('src_clf_loss',self.src_clf_loss.result()),
                                     ('fea_cor_mmd_loss',self.mmd_loss.result()),
                                     ('src_mmd_loss',self.src_mmd_loss.result()),
                               ("src_clf_acc", self.src_clf_acc.result()),
                               ("tar_clf_acc", self.tar_clf_acc.result()),
                               ("src_dc_acc", self.src_dc_acc.result()),
                               ("tar_dc_acc", self.tar_dc_acc.result())])
        return self.train_loss.result(),self.src_clf_acc.result()

    def train_one_epoch_end(self,x_s,y_s,C = 10):
        '''
        该方法用于计算源域类间距离，使得类间距离最大化，然后利用1-(1+e^-x)^-1函数式，将最大化类间距离转化为最小化损失函数
        :param x_s:
        :param y_s:
        :param C:
        :return:
        '''
        loss=0
        C_plus=C*(C-1)*1.0/2
        all_classes_samples = []
        for c in range(C):
            c_mask = y_s == c
            all_classes_samples.append(x_s[c_mask])
        for c1 in range(C):
            for c2 in range(c1+1,C):
                if all_classes_samples[c1].shape[0]!=0 and all_classes_samples[c2].shape[0]!=0 :
                    loss+=tf.reduce_mean(get_cos_dist(all_classes_samples[c1],all_classes_samples[c2]))
                else:
                    C_plus=C_plus-1
        res=loss/float(C_plus)
        return tf.exp(-res)/(1.0+tf.exp(-res)) #最大化该损失函数->最小化该损失函数

    def eval_one_epoch(self,t_iter,val_iter_num,ep):
        """
        这是一个周期的模型验证函数
        :param val_target_datagen: 目标域验证数据集生成器
        :param val_iter_num: 一个验证周期的迭代次数
        :param ep: 当前验证周期
        :return:
        """
        for i in np.arange(1, val_iter_num + 1):
            # 计算目标域数据的图像分类预测输出和域分类预测输出
            x_t, y_t, d_t = t_iter.__next__()
            tar_features = self.feature_extractor(x_t, training=False)
            tar_clf = self.clf_classifer(tar_features, training=False)

            # 计算目标域预测相关损失
            loss1 = self.loss(y_t,tar_clf)
            acc1=self.acc(y_t,tar_clf)
            self.acc.reset_states()
            target_loss = tf.reduce_mean(loss1)

            # 更新训练损失与训练精度
            self.val_loss(target_loss)
            self.val_acc(acc1)
        return self.val_loss.result(),self.val_acc.result()

#遍历所有可迁移对象
import numpy as np
compare_sets=(("A","B"),("B","A"),("A","C"),("C","A"),("D","A"),("A","D"),("B","C"),("C","B")("B","D"),("D","B"),("C","D"),("D","C"))
for compare_set in compare_sets:
    print(compare_set[0]+"->"+compare_set[1]+":")
    x_s=np.load('oral_data/'+compare_set[0]+'_data.npy')
    x_t=np.load('oral_data/'+compare_set[1]+'_data.npy')
    y_s=np.load('oral_data/'+compare_set[0]+'_label.npy')
    y_t=np.load('oral_data/'+compare_set[1]+'_label.npy')
    d_s=np.zeros_like(y_s)
    d_t=np.ones_like(y_t)
    x_s=np.reshape(x_s,newshape=(-1,32,32,1))
    x_t=np.reshape(x_t,newshape=(-1,32,32,1))
    s_sets=tf.data.Dataset.from_tensor_slices((x_s,y_s,d_s))
    t_sets=tf.data.Dataset.from_tensor_slices((x_t,y_t,d_t))
    model=WDCNN_plus()
    model.transfer_names=compare_set
    train_iter_num = min(int(len(x_s) // batch_size),int(len(x_t) // batch_size))
    val_iter_num =int(len(x_t) // batch_size)
    model.train(s_sets=s_sets,t_sets=t_sets,train_iter_num=train_iter_num,val_iter_num=val_iter_num)

