# 使用xception在视频上做检测
# 在图片文件夹上做test
from UltraSonicUtility.xcep.MyXception import *

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

        json_str = detect([scaled_frame])
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


def test_on_img_folder(imgFolder, errFolder):
    """imgFolder下的目錄必須是mappding中的值"""
    init()

    y_true = []
    y_pred = []
    y_props = []
    path = imgFolder
    for clsname in os.listdir(path):
        cls_path = os.path.join(path, clsname)
        for imgname in os.listdir(cls_path):
            if not imgname.endswith('.jpg'):
                continue
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

    return y_true,y_pred,y_props


if __name__ == '__main__':
    test_on_img_folder(r'D:\cls_images\sheared\test', r'D:\cls_images\sheared\error')
    # detect_on_video_folder(r'D:\testVideo',r'D:\testVideo\xcepimgs')
