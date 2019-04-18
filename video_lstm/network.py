import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.layers import LSTM,Dense,Conv2D,MaxPooling2D,Input,Flatten
from keras.models import Model
from keras.utils import plot_model
import numpy as np

# data_dim = 100  # 每张图片的最终特征值有100个
# timestaps = 10  # 设置每10张图片构成一个视频蒙太奇
# num_classes = 6  # 一共三个大类，6个类别
# img_width,img_height =227,227


def j_cnn(img_width=227, img_height=227,weights=None):
    img_width, img_height = img_width, img_height
    input = Input(shape=(img_height,img_width,1),dtype=np.float32)
    c1 = Conv2D(24,kernel_size=(11,11),strides=4,name='C1')(input)
    m1 = MaxPooling2D(pool_size=(3,3),strides=2,name='M1')(c1)
    c2 = Conv2D(24,kernel_size=(5,5),strides=2,padding='same',name='C2')(m1)
    m2 = MaxPooling2D(pool_size=(3,3),strides=2,padding='same',name='M2')(c2)
    c3 = Conv2D(24,kernel_size=(3,3),strides=1,padding='same',name='C3')(m2)
    c4 = Conv2D(24,kernel_size=(3,3),strides=1,padding='same',name='C4')(c3)
    c5 = Conv2D(24,kernel_size=(3,3),strides=1,padding='same',name='C5')(c4)
    m5 = MaxPooling2D(pool_size=(3,3),strides=2,name='M5')(c5)
    flatten = Flatten()(m5)
    f6 = Dense(100,activation='relu',name='F6')(flatten)
    f7 = Dense(6,activation='softmax',name='F7')(f6)

    model = Model(input=input,output = f7)
    if weights!=None:
        model.load_weights(weights)
    return model

if __name__ == '__main__':
    model = j_cnn()
    model.summary()
    model.compile()


