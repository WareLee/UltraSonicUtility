import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications import Xception
from keras.metrics import categorical_accuracy
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.layers import Conv2D, GlobalAveragePooling2D, Reshape,Softmax
from keras.models import Model, Sequential
import cv2

train_data_dir = r'D:\warelee\UltraAI\datasets\train\xception\train500'
validation_data_dir = r'D:\warelee\UltraAI\datasets\test\xception\test200'
cls_folder = ['hc', 'ac', 'fl', 'nhc', 'nac', 'nfl', 'bg']
classes = len(cls_folder)
img_height, img_width = 600, 600
batch_size = 4
nb_train_samples = 500 * classes
nb_validation_samples = 200 * classes
epochs = 50
weights = None
filepath = 'ep{epoch:03d}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.h5'
saving_path = r'D:\warelee\UltraAI\datasets\TrainModel\xception\conv_300300'
log_dir='../../tmp/log'


def _preprocessing_function(np_img):
    # 转换颜色空间
    tmp = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    # 归一化到（-1，1）
    return tmp.astype('float32') / 255.


# （图片大小不一致）data prepare ，先填充成正方形，再缩放到400x400
# resize and fill_constant, have to be same shape?
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='constant',
                             preprocessing_function=_preprocessing_function
                             )
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

# network
base_model = Xception(include_top=False, weights=weights, input_shape=(None, None, 3))
top_model = Sequential()
top_model.add(GlobalAveragePooling2D())
top_model.add(Reshape(target_shape=(1, 1, 2048)))
top_model.add(Conv2D(filters=classes, kernel_size=(1, 1)))
top_model.add(Reshape(target_shape=(classes,)))
top_model.add(Softmax())
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# plot
print(model.summary())
# plot_model(base_model, to_file='model3.png')
# complie
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train
board = TensorBoard(log_dir=log_dir,histogram_freq=0,embeddings_freq=0,write_images=True)
checkpoint = ModelCheckpoint(os.path.join(saving_path, filepath), monitor='val_loss', verbose=1, save_best_only=False,
                             save_weights_only=True,
                             mode='min', period=1)
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint,board])
