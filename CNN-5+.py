import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import datetime
from tensorflow.keras.utils import Progbar

batch_size=256
all_acc=[]
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

def block():
    return keras.Sequential([
        keras.layers.Conv2D(16,(3,3),1,'same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2,2),2),
        keras.layers.Conv2D(32,(3,3),1,'same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2,2),2),
        keras.layers.Conv2D(64,(3,3),1,'same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2,2),2),
        keras.layers.Conv2D(64,(3,3),1,'same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((2,2),2),
        keras.layers.Conv2D(64,(3,3),1,'same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.GlobalAveragePooling2D()
    ])

def classifer():
    return keras.Sequential([
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32),
        keras.layers.Activation('relu'),
        keras.layers.Dense(10)
    ])
def domain_classifer():
    return keras.Sequential([
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32),
        keras.layers.Activation('relu'),
        keras.layers.Dense(2)
    ])

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
        self.epoch =50

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
        self.src_clf_acc = keras.metrics.Mean("src_clf_acc", dtype=tf.float32)
        self.tar_clf_acc = keras.metrics.Mean("tar_clf_acc", dtype=tf.float32)
        self.src_dc_acc = keras.metrics.Mean("src_dc_acc", dtype=tf.float32)
        self.tar_dc_acc = keras.metrics.Mean("tar_dc_acc", dtype=tf.float32)
        self.val_loss =keras.metrics.Mean("val_loss", dtype=tf.float32)
        self.val_acc =keras.metrics.Mean("val_acc", dtype=tf.float32)
        # 定义优化器
        self.optimizer = tf.keras.optimizers.SGD(self.init_learningrate,
                                                 momentum=0.7)

        '''
        # 初始化早停策略
        self.early_stopping = EarlyStopping(min_delta=1e-5, patience=100, verbose=1)
        '''

    def build_DANN(self):
        """
        这是搭建域适配网络的函数
        :return:
        """
        # 定义源域、目标域的图像输入和DANN模型图像输入
        self.feature_extractor = block()
        self.clf_classifer = classifer()
        self.dc_classifer = domain_classifer()
        self.grl = GRL()
        src_inputs=keras.Input(shape=(32,32,1))
        tar_inputs=keras.Input(shape=(32,32,1))
        src_fea=self.feature_extractor(src_inputs)
        tar_fea=self.feature_extractor(tar_inputs)
        src_dc=self.dc_classifer(self.grl(src_fea))
        tar_dc=self.dc_classifer(self.grl(tar_fea))
        src_clf=self.clf_classifer(src_fea)
        tar_clf=self.clf_classifer(src_fea)
        self.dann_model = keras.Model(inputs=[src_inputs,tar_inputs],outputs=[src_clf,tar_clf,src_dc,tar_dc])
        self.dann_model.summary()
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
        all_acc=[]
        print(time)
        self.progbar = Progbar(self.epoch+1)
        for ep in np.arange(1,self.epoch+1,1):
            # 初始化精度条
            self.progbar.update(ep)
            # print('Epoch {}/{}'.format(ep, self.epoch))
            s_iter=iter(s_sets.batch(batch_size))
            t_iter=iter(t_sets.batch(batch_size))
            val_iter = iter(t_sets.batch(batch_size))
            # 进行一个周期的模型训练
            train_loss,src_clf_acc = self.train_one_epoch\
                (s_iter,t_iter,train_iter_num,ep)
            # 进行一个周期的模型验证
            val_loss,val_acc = self.eval_one_epoch(val_iter,val_iter_num,ep)
            # 更新进度条
            # self.progbar.update(train_iter_num+1, [('val_loss', self.val_loss.result()),
                                                   #("val_acc", self.val_acc.result())])
            # 保存训练过程中的模型
            str = "Epoch{:03d}-train_loss-{:.3f}-val_loss-{:.3f}-src_acc-{:.3f}-val_acc-{:.3f}" \
                .format(ep, self.train_loss.result(), self.val_loss.result(), self.src_clf_acc.result(), self.val_acc.result())
            all_acc.append(self.val_acc.result())
            # print(str)
            # 损失和指标清零
            self.train_loss.reset_states()
            self.src_clf_loss.reset_states()
            self.src_dc_loss.reset_states()
            self.tar_dc_loss.reset_states()
            self.src_clf_acc.reset_states()
            self.tar_clf_acc.reset_states()
            self.src_dc_acc.reset_states()
            self.tar_dc_acc.reset_states()
            self.val_acc.reset_states()
            self.val_loss.reset_states()
            '''
            # 判断是否需要早停模型训练过程，判断指标为目标域的图像分类精度
            stop_training = self.early_stopping.on_epoch_end(ep, val_image_cls_acc)
            if stop_training:
                break
            '''

            time = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            print(time)
        # print('\n----------- end to train -----------\n')
        if ep==self.epoch:
            all_acc = np.array(all_acc)
            all_acc = np.sort(all_acc)
            acc = all_acc[-10:]
            self.progbar.update(self.epoch+1,[("mean",np.mean(acc)),("hurt",(np.max(acc) - np.mean(acc)))])

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
                tar_features = self.feature_extractor(x_t,training=True)
                src_dc = self.dc_classifer(self.grl(src_features, self.grl_lambd),training=True)
                tar_dc = self.dc_classifer(self.grl(tar_features, self.grl_lambd),training=True)
                src_clf = self.clf_classifer(src_features,training=True)
                tar_clf = self.clf_classifer(tar_features,training=True)

                loss1=self.loss(y_s,src_clf)
                loss2=self.loss(d_s,src_dc)
                loss3=self.loss(d_t,tar_dc)
                acc1=self.acc(y_s,src_clf)
                self.acc.reset_states()
                acc2=self.acc(y_t,tar_clf)
                self.acc.reset_states()
                acc3=self.acc(d_s,src_dc)
                self.acc.reset_states()
                acc4=self.acc(d_t,tar_dc)
                self.acc.reset_states()

                # 计算训练损失、图像分类精度和域分类精度
                loss = tf.reduce_mean(loss1) +(tf.reduce_mean(loss2)+tf.reduce_mean(loss3))
            # 自定义优化过程
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            # 计算平均损失与精度
            self.train_loss(loss)
            self.src_clf_loss(loss1)
            self.src_dc_loss(loss2)
            self.tar_dc_loss(loss3)
            self.src_clf_acc(acc1)
            self.tar_clf_acc(acc2)
            self.src_dc_acc(acc3)
            self.tar_dc_acc(acc4)

            # 更新进度条
            # self.progbar.update(i, [('loss', self.train_loss.result()),
            #                    ("src_clf_acc", self.src_clf_acc.result()),
            #                    ("tar_clf_acc", self.tar_clf_acc.result()),
            #                    ("src_dc_acc", self.src_dc_acc.result()),
            #                    ("tar_dc_acc", self.tar_dc_acc.result())])

        return self.train_loss.result(),self.src_clf_acc.result()

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
        return self.val_loss.result(), self.val_acc.result()

import numpy as np
compare_sets=(("A","B"),("B","A"),("A","C"),("C","A"),("A","D"),("D","A"),("B","C"),("C","B"),("B","D"),("D","B"),("C","D"),("D","C"))
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
    # model.build_DANN()
    train_iter_num = int(len(x_s) // batch_size)
    val_iter_num = train_iter_num
    model.train(s_sets=s_sets,t_sets=t_sets,train_iter_num=train_iter_num,val_iter_num=val_iter_num)
    # model.dann_model.save("wdcnn_dann.h5")


