# 使用Xception 训练超声数据， 分两类标准和非标准（3个部位*2细分类）
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.applications import Xception
from keras.utils import generic_utils
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

train_data_dir = r'D:\warelee\datasets\train\xception\train'
cls_folder=['hc','ac','fl','nhc','nac','nfl','bg']
# validation_data_dir = r'D:\cls_images'
img_width, img_height = 800, 600
batch_size = 4
nb_train_samples = 2500*7
# nb_validation_samples = 2000
epochs = 50
steps = nb_train_samples // batch_size
saving_path = r'D:\warelee\datasets\TrainModel\xception\nfl'
classes=7
weights =os.path.join(r'D:\warelee\datasets\TrainModel\xception\nfl','epoch4-train_loss0.160-train_acc1.000.h5')
# weights=None
# network
model = Xception(include_top=True, weights=weights, input_shape=(img_height, img_width, 3), classes=classes)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# data prepare
datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=10,
                             width_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='constant'
                             )
train_generator = datagen.flow_from_directory(train_data_dir,
                                              classes=cls_folder,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')
print(train_generator.class_indices)

# validation_generator = datagen.flow_from_directory(validation_data_dir,
#                                                    target_size=(img_width, img_height),
#                                                    batch_size=batch_size,
#                                                    shuffle=True, class_mode='categorical')

# train
for epoch in range(1, epochs + 1):
    print('Epoch: ', epoch)
    progbar = generic_utils.Progbar(nb_train_samples)
    for i in range(1, steps + 1):
        x_batch, y_batch = train_generator.next()
        x_batch = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in x_batch])
        loss, train_acc = model.train_on_batch(x_batch, y_batch)
        progbar.add(x_batch.shape[0], values=[('train loss', loss), ('train acc', train_acc)])
    # saving model
    m_name = 'epoch{}-train_loss{:.3f}-train_acc{:.3f}.h5'.format(epoch, loss, train_acc)
    print('Saving ', m_name)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    model.save(os.path.join(saving_path, m_name))
