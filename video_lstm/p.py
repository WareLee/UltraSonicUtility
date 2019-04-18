import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
import cv2
from video_lstm.network import j_cnn

train_data_dir = r'F:\workspace\ultrasonic\hnuMedical2\ImageWare\grouped_660660\train500_1'
validation_data_dir = r'F:\workspace\ultrasonic\hnuMedical2\ImageWare\grouped_660660\test200_1'
cls_folder = ['hc', 'ac', 'fl', 'nhc', 'nac', 'nfl', 'bg']
classes = len(cls_folder)
img_height, img_width = 227, 227
batch_size = 4
nb_train_samples = 500 * classes
nb_validation_samples = 200 * classes
epochs = 50
weights = None
filepath = 'ep{epoch:03d}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.h5'
# saving_path = r'D:\warelee\UltraAI\datasets\TrainModel\xception\conv_300300'
saving_path = r''
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
model = j_cnn(img_width,img_height,weights=weights)
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
