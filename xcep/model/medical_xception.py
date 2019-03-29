# 使用Xception 训练超声数据， 分两类标准和非标准（3个部位*2细分类）
# TODO 数据的预处理来适应Xception的输入
# TODO 在三类上训练
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from keras.applications import Xception
from keras.metrics import categorical_accuracy
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

train_data_dir = r'../cls_imgs_sheared'
validation_data_dir = r'../cls_imgs_sheared_validation'
img_width, img_height = 800, 600
batch_size = 4
nb_train_samples = 14000
nb_validation_samples = 14000
epochs = 50
classes = 7
weights = os.path.join('./model_xcep_880x660_with_bg', 'model-ep007-loss0.2806-val_acc0.9626.h5')
filepath = 'model_xcep_880x660_with_bg/model-ep{epoch:03d}-loss{loss:.4f}-val_acc{val_categorical_accuracy:.4f}.h5'

datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=10,
                             width_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='constant')
model = Xception(include_top=True, weights=weights, input_shape=(img_height, img_width, 3), classes=classes)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
# plot_model(model,'xception_top_cls6.png')
train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')
print(train_generator.class_indices)
validation_generator = datagen.flow_from_directory(validation_data_dir,
                                                   target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   shuffle=True, class_mode='categorical')

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,
                             mode='min', period=1)
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint])
