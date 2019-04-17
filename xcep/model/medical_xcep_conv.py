# TODO  1.Xception conv化，并使用rgb图像训练
# TODO  2.Xception 再bbox上训练
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications import Xception
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, GlobalAveragePooling2D, Reshape
from keras.models import Model, Sequential

train_data_dir = r'../cls_imgs'
validation_data_dir = r'../cls_imgs_sheared_test_std'
batch_size = 4
nb_train_samples = 12000
nb_validation_samples = 6000
epochs = 50
classes = 7
weights = None
filepath = 'model_xcep_880x660/model-ep{epoch:03d}-loss{loss:.4f}-val_acc{val_categorical_accuracy:.4f}.h5'

# （图片大小不一致）data prepare ，先填充成正方形，再缩放到400x400
# resize and fill_constant, have to be same shape?
datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=10,
                             # width_shift_range=0.1,
                             # shear_range=0.1,
                             # zoom_range=0.1,
                             vertical_flip=True,
                             horizontal_flip=True,
                             fill_mode='constant')
train_generator = datagen.flow_from_directory(train_data_dir,
                                              # target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')
print(train_generator.class_indices)
validation_generator = datagen.flow_from_directory(validation_data_dir,
                                                   # target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   shuffle=True, class_mode='categorical')

# network
base_model = Xception(include_top=False, weights=weights, input_shape=(None, None, 3))
top_model = Sequential()
top_model.add(GlobalAveragePooling2D())
top_model.add(Reshape(target_shape=(1, 1, 2048)))
top_model.add(Conv2D(filters=7, kernel_size=(1, 1)))
top_model.add(Reshape(target_shape=(classes,)))
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# plot
print(model.summary())
plot_model(base_model, to_file='model3.png')
# complie
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
# train
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,
                             mode='min', period=1)
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint])
