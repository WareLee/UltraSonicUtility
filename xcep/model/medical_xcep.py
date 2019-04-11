# 使用Xception 训练超声数据， 分两类标准和非标准（3个部位*2细分类）
# TODO 数据的预处理来适应Xception的输入
# TODO 在三类上训练
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.applications import Xception
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
import cv2
import os

train_data_dir = r'D:\warelee\datasets\train\xception\train500'
validation_data_dir =r'D:\warelee\datasets\test\xception\test200'
cls_folder=['hc','ac','fl','nhc','nac','nfl','bg']
img_width, img_height = 800, 600
nb_train_samples = 500*7
nb_validation_samples= 200*7
batch_size = 4
epochs = 50
classes=7
steps = nb_train_samples // batch_size
saving_path = r'D:\warelee\datasets\TrainModel\xception\nfl_500'
weights =os.path.join(r'D:\warelee\datasets\TrainModel\xception\nfl_500','ep002-val_loss0.3871-val_acc0.9136.h5')


def _preprocessing_function(np_img):
    # 转换颜色空间
    tmp = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    # 归一化到（-1，1）
    return tmp / 127. - 1.


datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='constant',
                             preprocessing_function=_preprocessing_function
                             )
model = Xception(include_top=True, weights=weights, input_shape=(img_height, img_width, 3), classes=classes)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(model,'xception_top_cls6.png')
train_generator = datagen.flow_from_directory(train_data_dir,
                                              classes=cls_folder,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')

print(train_generator.class_indices)
validation_generator = datagen.flow_from_directory(validation_data_dir,
                                                   classes=cls_folder,
                                                   target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   shuffle=True, class_mode='categorical')

filepath = 'ep{epoch:03d}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(os.path.join(saving_path,filepath), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,
                             mode='min', period=1)
board = TensorBoard(log_dir='../../tmp/log',histogram_freq=0,embeddings_freq=0,write_images=True)
early_stoping = EarlyStopping(monitor='acc',patience=3)
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[board,checkpoint,early_stoping])
