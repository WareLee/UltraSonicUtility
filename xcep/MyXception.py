# 该脚本跑出来的结果和训练中展现的不一致（nfl被分到了fl中）
from keras.applications import Xception

import cv2
import os
import numpy as np
import time
import json

MODEL_PATH = os.path.join(r'D:\warelee\datasets\TrainModel\xception', 'epoch2-train_loss0.000188940524822101-train_acc1.0.h5')
# mapping = {-1: 'others', 0: 'ac', 1: 'fl', 2: 'hc', 3: 'nac', 4: 'nfl', 5: 'nhc'}
# std2nstd_map = {0: 3, 1: 4, 2:5 }
mapping = {0: 'ac', 1: 'bg', 2: 'fl', 3: 'hc', 4: 'nac', 5: 'nfl', 6: 'nhc'}
std2nstd_map = {0: 4, 2: 5, 3: 6}
# 配合c++ application
incapp = False
map2cmap = {0: 1, 2: 2, 3: 0, 4: 4, 5: 5, 6: 3, 1: -1}

thresholde = 0.7
log_path = 'evaluate_sheared.log'
# image shape of network input
input_height, input_width = 600, 800
classes = 7

def init():
    model_path = MODEL_PATH

    # load Xcep model
    global model
    model = Xception(include_top=True, weights=model_path, input_shape=(input_height, input_width, 3), classes=classes)

    print('Xception loading ... ')


def detect(imageList):
    """使用Xception推理,images input in bgr,rescaled and resized"""
    if len(imageList) == 0:
        return

    # process image
    start = time.time()

    if incapp:
        # cvt to bgr and rescale value domain
        # TODO use only c++ application, should be deleted when run this script only
        # imageList = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) / 255.0 for img in imageList]
        imageList = [img / 255.0 for img in imageList]

    images = np.array(imageList)
    print('images.shape : ', images.shape)

    global model
    # TODO 转成model.predict会不会快一点
    xcep_prop = model.predict_on_batch(images)
    max_ind = np.argmax(xcep_prop, axis=1)
    max_ind = [int(i) for i in max_ind]  # convert to int32
    props = []  # 各个样本的最佳类别对应的概率值
    for hang, ind in enumerate(max_ind):
        props.append(xcep_prop[hang][ind])

    print('max_ind : ', max_ind)
    print('props : ', props)
    print("processing time: ", time.time() - start)

    # 整合结果---》
    result = []
    for ind, prop in zip(max_ind, props):
        record = {}
        record['prop'] = float(prop)
        if int(ind) in std2nstd_map.keys() and prop < thresholde:
            # todo
            ind = std2nstd_map[int(ind)]
            if incapp:
                record['label'] = map2cmap[ind]
            else:
                record['label'] = ind
            record['score'] = round(float(prop), 4) * (-1)
            # result.append('' + mapping[int(ind)] + ':' + str(round(float(prop), 4) * (-1)))
        else:
            if incapp:
                record['label'] = map2cmap[int(ind)]
            else:
                record['label'] = int(ind)
            record['score'] = float(prop)
        result.append(record)
        # result.append('' + mapping[int(ind)] + ':' + str(round(float(prop), 4)))

    json_str = json.dumps(result)
    print(json_str)

    return json_str


if __name__ == '__main__':
    init()
    path = r'D:\test_imgs_sheared_1000\imgs\ac'
    img_batch = []
    i = 0
    for name in os.listdir(path):
        img = cv2.imread(os.path.join(path, name))
        img = cv2.resize(img, dsize=(800, 600))
        img_batch.append(img)
        i += 1
        if i >= 8:
            detect(img_batch)
            img_batch = []
            i = 0
        else:
            continue
