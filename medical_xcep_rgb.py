# 使用rgb的格式读取图片
import os
import cv2
import numpy as np
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.metrics import categorical_accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def load_medical_data(path, dsize=(800, 600), cls_mapping=None):
    # generate a folder to cls mapping
    if cls_mapping == None:
        cls_mapping = {}
        cls_num = 0
        for cls in os.listdir(path):
            cls_folder = os.path.join(path, cls)
            if os.path.isdir(cls_folder):
                cls_mapping[cls] = cls_num
                cls_num += 1
    # load all data
    x_es, y_es = [], []
    for cls in cls_mapping.keys():
        cls_path = os.path.join(path, cls)
        for imgname in os.listdir(cls_path):
            if imgname.endswith('.jpg'):
                x_es.append(cv2.resize(cv2.imread(os.path.join(cls_path, imgname)), dsize=dsize))
                y_es.append([cls_mapping[cls]])

    return np.array(x_es), np.array(y_es), cls_mapping


if __name__ == '__main__':
    batch_size = 1
    epochs = 50
    # steps = 2500
    weights = None
    img_d_height, img_d_width = 600, 800
    saving_path = r'./'

    # TODO Currently , it did not be taken into account if imgset is to big to load in one time
    # TODO Add test set for validation
    # train data prepare
    path = r'D:\tmp\tmp'
    x_train, y_train, cls_mapping = load_medical_data(path, dsize=(img_d_width, img_d_height))
    y_train = np_utils.to_categorical(y_train, num_classes=len(cls_mapping.keys()))
    print(x_train.shape)
    print(y_train.shape)
    print(cls_mapping)

    # generator
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True,
                                 fill_mode='constant'
                                 )

    # network
    model = Xception(include_top=True, weights=weights, input_shape=(img_d_height, img_d_width, 3),
                     classes=len(cls_mapping.keys()))
    print(model.summary())

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])

    # train
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        progbar = generic_utils.Progbar(x_train.shape[0])
        steps = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True):
            loss, train_acc = model.train_on_batch(x_batch, y_batch)
            steps += x_batch.shape[0]
            progbar.add(x_batch.shape[0], values=[('train loss', loss), ('train acc', train_acc)])
            if steps >= x_train.shape[0]:
                break
        # saving model
        m_name = 'epoch%d-train_loss%.3f-train_acc%.3f.h5' % (epoch, loss, train_acc)
        print('Saving ', m_name)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
        model.save(os.path.join(saving_path, m_name))
