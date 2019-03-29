# 使用Xception 训练超声数据， 分两类标准和非标准（3个部位*2细分类）
# TODO 数据的预处理来适应Xception的输入
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.applications import Xception
from keras.models import Model
from keras.metrics import categorical_accuracy
# from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import cv2


test_data_dir = r'D:\test_imgs_sheared_1000\imgs'
img_width, img_height = 800, 600
batch_size = 1
nb_test_samples = 2100
# mapping = {-1:'others',0: 'ac', 1:'fl',  2:'hc', 3:'nac', 4:'nfl',  5:'nhc'}
mapping = {0: 'ac', 1: 'bg', 2: 'fl', 3: 'hc', 4: 'nac', 5: 'nfl', 6: 'nhc'}
log_path='validation_xcep2.log'
error_path='D:\cls_images\sheared\error'
MODEL_PATH = os.path.join(r'D:\cls_images\sheared\TrainModel', 'epoch1-train_loss0.006963605992496014-train_acc1.0.h5')
# MODEL_PATH = None
classes=7

y_true=[]
y_pred =[]
props =[]

datagen = ImageDataGenerator(rescale=1. / 255)
model = Xception(include_top=True, weights=MODEL_PATH, input_shape=(img_height, img_width, 3), classes=classes)
print(model.summary())

test_generator = datagen.flow_from_directory(test_data_dir,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True, class_mode='categorical')
print(test_generator.class_indices)
i = 0

for batch in test_generator:
    val_x,val_y=batch

    true_ind = int(list(np.argmax(val_y,axis=1))[0])
    res = model.predict_on_batch(val_x)
    pred_ind = int(list(np.argmax(res,axis=1))[0])
    print(res)
    print(pred_ind)

    # TODO pred_ind應該是個list的
    if pred_ind in {0,2,3}:
        props.append(1-res[0][pred_ind])

    # store pred error imgs
    if true_ind!=pred_ind:
        cls_err_path = os.path.join(error_path,mapping[true_ind])
        if not os.path.exists(cls_err_path):
            os.mkdir(cls_err_path)
        uint8_img = cv2.normalize(val_x[0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(cls_err_path,str(i)+'_'+mapping[true_ind]+'vs'+mapping[pred_ind]+'.jpg'),uint8_img)

    y_true.append(mapping[true_ind])
    y_pred.append(mapping[pred_ind])
    # print('{} v.s {}'.format(true_ind,pred_ind))

    if i>=nb_test_samples:
        break
    if i % 100==0:
        print('Num of detected images: ',i)
    i+=1

print(props)
accu = accuracy_score(y_true, y_pred)
cfm = confusion_matrix(y_true,y_pred)
clsrep = classification_report(y_true, y_pred)
print(accu)
print()
print(cfm)
print()
print(clsrep)
with open(os.path.join(error_path,log_path),'w',encoding='utf-8') as f:
    f.write(str(float(accu)))
    f.write('\n')
    f.write(str(cfm))
    f.write('\n')
    f.write(clsrep)
