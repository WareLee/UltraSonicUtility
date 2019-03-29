# 使用Xception 训练超声数据， 分两类标准和非标准（3个部位*2细分类）
# TODO 数据的预处理来适应Xception的输入
# TODO 在三类上训练
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.applications import Xception
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

train_data_dir = r'D:\cls_images'
validation_data_dir = r'D:\cls_images'
img_width, img_height = 800, 600
batch_size = 4
nb_train_samples = 2000
nb_validation_samples = 2000
epochs = 50

datagen = ImageDataGenerator(rescale=1. / 255)
model = Xception(include_top=True, weights=None, input_shape=(img_width, img_height, 3), classes=6)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(model,'xception_top_cls6.png')
train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')

print(train_generator.class_indices)
validation_generator = datagen.flow_from_directory(validation_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   shuffle=True, class_mode='categorical')

filepath = 'little-ep{epoch:03d}-loss{loss:.4f}-val_acc{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True,
                             mode='max', period=2)
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint])
