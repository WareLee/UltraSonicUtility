# 该脚本跑出来的结果和训练中展现的不一致（nfl被分到了fl中）
from keras.applications import Xception
from keras.preprocessing.image import load_img, save_img

import cv2
import os
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_PATH = os.path.join(r'D:\cls_images\sheared\TrainModel', 'epoch2-train_loss0.000188940524822101-train_acc1.0.h5')
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

def detect_from_video(video_path, target_path, visualize=False):
    """detect frame by frame"""
    # TODO cv2和keras.load_img读取的图片rgb vs bgr ,值也不同
    reader = cv2.VideoCapture()
    if not reader.open(video_path):
        print('Video can not be opened: ' + video_path)
        return []
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    date = os.path.basename(os.path.dirname(video_path))

    name_prefix = date + '_' + video_name

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur_id = 0
    cur_frame = np.zeros([height, width, 3], np.uint8)

    while reader.read(cur_frame):
        # bgr
        bgr_img = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)
        # resize and rescaled;
        scaled_frame = cv2.resize(bgr_img, dsize=(input_width, input_height)) / 255.0

        json_str = detect(scaled_frame)
        xcep_res = json.loads(json_str)

        cur_frame_name = name_prefix + '_' + str(cur_id) + '.jpg'

        if visualize:
            cv2.imshow(cur_frame_name + ':' + json_str, scaled_frame)
            cv2.waitKey()

        # store frame by classifications
        clsname = mapping[int(xcep_res[0]['label'])]
        cls_path = os.path.join(target_path, clsname)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)
        cv2.imwrite(os.path.join(cls_path, cur_frame_name), cur_frame)

        cur_id += 1

        # skip one frame
        if not reader.grab():
            break
    reader.release()

def detect_from_video2(video_path, target_path, visualize=False,batch_size=1):
    """detect frame by frame"""
    # TODO cv2和keras.load_img读取的图片rgb vs bgr ,值也不同
    reader = cv2.VideoCapture()
    if not reader.open(video_path):
        print('Video can not be opened: ' + video_path)
        return []
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    date = os.path.basename(os.path.dirname(video_path))

    name_prefix = date + '_' + video_name

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur_id = 0
    cur_frame = np.zeros([height, width, 3], np.uint8)
    caped_frames =[]
    scaled_frames =[]

    while reader.read(cur_frame):
        caped_frames.append(cur_frame)
        # bgr
        bgr_img = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)
        # resize and rescaled;
        scaled_frame = cv2.resize(bgr_img, dsize=(input_width, input_height)) / 255.0
        scaled_frames.append(scaled_frame)
        cur_id += 1

        if len(scaled_frames)>=batch_size:
            json_str = detect(scaled_frames)
        else:
            continue
        scaled_frames=[]
        xcep_res = json.loads(json_str)
        print(xcep_res)

        # cur_frame_name = name_prefix + '_' + str(cur_id) + '.jpg'
        #
        # if visualize:
        #     cv2.imshow(cur_frame_name + ':' + json_str, scaled_frame)
        #     cv2.waitKey()
        #
        # # store frame by classifications
        # clsname = mapping[int(xcep_res[0]['label'])]
        # cls_path = os.path.join(target_path, clsname)
        # if not os.path.exists(cls_path):
        #     os.makedirs(cls_path)
        # cv2.imwrite(os.path.join(cls_path, cur_frame_name), cur_frame)

        caped_frames=[]
        # skip one frame
        if not reader.grab():
            break
    reader.release()


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
        imageList = [img/ 255.0 for img in imageList]

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
        record ={}
        record['prop']=float(prop)
        if int(ind) in std2nstd_map.keys() and prop < thresholde:
            # todo
            ind = std2nstd_map[int(ind)]
            if incapp:
                record['label']=map2cmap[ind]
            else:
                record['label'] = ind
            record['score']= round(float(prop), 4) * (-1)
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


def detect_on_video_folder(videoFolder, targetPath, visualize=False):
    init()
    avis = []
    for avi in os.listdir(videoFolder):
        if avi.endswith('.avi'):
            avis.append(os.path.join(videoFolder, avi))
    print('Avis will be detected. --> ', avis)
    print('Frames detected will stored in : ', targetPath)
    for avi in avis:
        detect_from_video(avi, targetPath, visualize=visualize)

def for_roc(y_true,y_pred,y_props):
    # y_true.append(clsname)
    # y_pred.append(pred_label)
    # y_props.append(prop)
    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    std = {'nac':'ac','nfl':'fl','nhc':'hc'}
    props ={}
    labels ={}
    for v in std.values():
        props[v]=[]
        labels[v]=[]

    for i,pred in enumerate(y_pred):
        if pred=='bg':
            continue
        # 将预测为非标准的对应的概率值替换为（1-非标准概率）
        if pred in std.keys():
            y_props[i]=1-y_props[i]
            y_pred[i]=std[pred]

        props[y_pred[i]].append(y_props[i])
        labels[y_pred[i]].append(y_true[i])

    for v in props.keys():
        fpr, tpr, thresholds = metrics.roc_curve(labels[v], props[v], pos_label=std.values())
        print('------------------------------')
        print(v)
        print(fpr)
        print(tpr)
        print(thresholds)
        plt.plot(fpr, tpr, marker='o')
        plt.show()

def test_on_img_folder(imgFolder, errFolder):
    """imgFolder下的目錄必須是mappding中的值"""
    init()

    y_true = []
    y_pred = []
    y_props =[]
    path = imgFolder
    for clsname in os.listdir(path):
        cls_path = os.path.join(path, clsname)
        for imgname in os.listdir(cls_path):
            if not imgname.endswith('.jpg'):
                continue
            # TODO 实验
            # img = cv2.imread(os.path.join(cls_path, imgname))
            # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            # bgr
            img = load_img(os.path.join(cls_path, imgname))
            # resized and rescaled
            img = cv2.resize(np.array(img), dsize=(input_width, input_height)) / 255.0
            json_str = detect([img])
            res_list = json.loads(json_str)
            pred_label = mapping[int(res_list[0]['label'])]
            prop = res_list[0]['prop']

            y_true.append(clsname)
            y_pred.append(pred_label)
            y_props.append(prop)

            # store error
            if pred_label != clsname:
                error_path = os.path.join(errFolder, clsname, 'error')
                if not os.path.exists(error_path):
                    os.makedirs(error_path)
                extra = json.loads(json_str)
                extra = mapping[extra[0]['label']] + str(extra[0]['score'])

                err_img_name = os.path.splitext(imgname)[0] + extra + '.jpg'
                save_img(os.path.join(error_path, err_img_name), img)

    accu = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    clsrep = classification_report(y_true, y_pred)
    with open(os.path.join(errFolder, log_path), 'w', encoding='utf-8') as f:
        f.write(str(float(accu)))
        f.write('\n')
        f.write(str(cfm))
        f.write('\n')
        f.write(clsrep)
    print(accu)
    print()
    print(cfm)
    print()
    print(clsrep)

    for_roc(y_true,y_pred,y_props)



if __name__ == '__main__':
    test_on_img_folder(r'D:\cls_images\sheared\test', r'D:\cls_images\sheared\error')
    # detect_on_video_folder(r'D:\testVideo',r'D:\testVideo\xcepimgs')

    # init()
    # path=r'D:\test_imgs_sheared_1000\imgs\ac'
    # img_batch =[]
    # i =0
    # for name in os.listdir(path):
    #     img = cv2.imread(os.path.join(path, name))
    #     img = cv2.resize(img,dsize=(800,600))
    #     img_batch.append(img)
    #     i+=1
    #     if i>=8:
    #         detect(img_batch)
    #         img_batch =[]
    #         i=0
    #     else:
    #         continue
