# -*- coding: utf-8 -*-
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.applications import Xception
from keras.utils import plot_model
import cv2
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(r'F:\workspace\TrainModel\nfl', 'epoch4-train_loss0.160-train_acc1.000.h5')
mapping = {0: 'hc', 1: 'ac', 2: 'fl', 3: 'nhc', 4: 'nac', 5: 'nfl', 6: 'bg'}
thresholde = 0.7
input_height, input_width = 600, 800
classes = 7


def load_roginal_img(img_path, dsize=(800, 600)):
    im_org = cv2.imread(img_path)
    img = cv2.resize(im_org, dsize=dsize)
    img = img / 255.0
    img_batch = np.array([img])
    plt.subplot(211)
    plt.imshow(img)
    plt.show()
    return img_batch


def load_xcep_model(img_batch):
    model = Xception(include_top=True, weights=MODEL_PATH, input_shape=(input_height, input_width, 3), classes=classes)
    model.summary()
    plot_model(model, 'xcep_model.jpg')
    # preds = model.predict(img_batch)
    return model, img_batch


def extract_features(ins, lyer, filters, layer_num, lyer_ind):
    '''
    提取指定模型指定层指定数目的feature map并输出到一幅图上.

    :param ins: tuple , 模型实例
    :param lyer: int or str , 提取指定层特征
    :param filters: 每层提取的feature map数
    :param layer_num: 一共提取多少层feature map
    :param lyer_ind: lyer所指层在所有要提取的层中是第几个
    :return: None
      '''
    if len(ins) != 2:
        print('parameter error:(model, instance)')
        return None
    model = ins[0]
    x = ins[1]
    if type(lyer) == type(1):
        model_extractfeatures = Model(input=model.input, output=model.get_layer(index=lyer).output)
    else:
        model_extractfeatures = Model(input=model.input, output=model.get_layer(name=lyer).output)
    fc2_features = model_extractfeatures.predict(x)

    if filters > fc2_features.shape[-1]:
        print('layer number error.', fc2_features.shape[-1], ',', filters)
        return None
    for i in range(filters):
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(filters, layer_num, lyer_ind + 1 + i * layer_num)
        plt.axis("off")
        if i < fc2_features.shape[-1]:
            plt.imshow(fc2_features[0, :, :, i])


def extract_features_batch(layers, model, filters):
    '''
       批量提取特征
       :param layers: int or list[str,] ,需要可视化的层
       :param model: 模型
       :param filters: feature map数
       :return: None
       '''

    # 有多少层需要被可视化
    layers_count = 0
    if isinstance(layers, list):
        layers_count = len(layers)
    else:
        layers_count = layers
        layers = [i for i in range(layers_count)]
    plt.figure(figsize=(filters, layers_count))
    plt.subplot(filters, layers_count, 1)
    for lyer in layers:
        extract_features(model, lyer, filters, layers_count, layers.index(lyer))
    plt.savefig('layers{}_vis.jpg'.format(lyer))
    plt.show()


if __name__ == '__main__':
    img_p = os.path.join(r'C:\Users\WareLee\Desktop\test\tmp',
                         '2019-02-12_08-48-11_14717.jpg')
    img_batch = load_roginal_img(img_p)
    ins = load_xcep_model(img_batch)
    layers = ['block1_conv1_act', 'block1_conv2_act', 'add_1', 'add_2', 'add_3']
    layers_2 = ['add_4', 'add_5', 'add_6', 'add_7',
              'add_8']
    layers_3 =['add_9','add_10', 'add_11', 'add_12', 'block14_sepconv2_act']
    extract_features_batch(layers, ins, 10)
    extract_features_batch(layers_2, ins, 10)
    extract_features_batch(layers_3, ins, 10)
