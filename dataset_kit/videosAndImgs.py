"""shape read is 1000x750 defaultly"""
import cv2
import os
import numpy as np
import shutil


def get_english_cls_name(clsname):
    if clsname.find('其他') >= 0:
        clsname = 'others'
    elif clsname.find('腹部') >= 0:
        if clsname.find('非标准') >= 0:
            clsname = 'nac'
        else:
            clsname = 'ac'
    elif clsname.find('股骨') >= 0:
        if clsname.find('非标准') >= 0:
            clsname = 'nfl'
        else:
            clsname = 'fl'
    elif clsname.find('丘脑') >= 0:
        if clsname.find('非标准') >= 0:
            clsname = 'nhc'
        else:
            clsname = 'hc'
    else:
        clsname='dontcare'

    return clsname

def sampling(src_folder,dst_folder,sep=2):
    """
    每间隔指定的图片采取一张
    :param src_folder:
    :param dst_folder:
    :param sep:
    :return:
    """
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    imgnames = os.listdir(src_folder)
    for i in range(0,len(imgnames),sep):
        print(imgnames[i])
        if imgnames[i].endswith('.jpg'):
            shutil.copy(os.path.join(src_folder,imgnames[i]),os.path.join(dst_folder,imgnames[i]))


def extract_imgs_from_vido(video_path, label_path, target_path, encoding='gbk'):
    """提取带标注文件的（单个）视频中的图片，到指定目录下，分门别类的存放

    :param video_path:   视频路径名
    :param label_path:   视频对应的标注文件名
    :param target_path:  提取出来的图片存放位置
    :param encoding:
    :return:
    """

    reader = cv2.VideoCapture()
    if not reader.open(video_path):
        print('Video can not be opened: ' + video_path)
        return []

    # 带后缀的name
    video_name = os.path.basename(video_path)
    # 不带后缀的name
    video_name = os.path.splitext(video_name)[0]
    # 要保存的图片名的前面部分
    name_prefix = video_name

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur_id = 1
    cur_frame = np.zeros([height, width, 3], np.uint8)

    # 读取视频对应的标注信息
    lines = []
    with open(label_path, encoding=encoding) as f:
        lines = f.readlines()
    while reader.read(cur_frame):
        # 当前图片名
        cur_frame_name = name_prefix + '_' + str(cur_id) + '.jpg'
        # 当前图片名,标注的类别(转为非中文)
        labeled_imgname, clsname = lines[cur_id - 1].strip().split(' ')
        clsname = get_english_cls_name(clsname)

        if cur_frame_name.strip() != labeled_imgname:
            print('Video frame name does not match labeled img name: {} v.s {}'.format(cur_frame_name, labeled_imgname))
            exit()
        # 生成对应类别存放目录
        cls_path = os.path.join(target_path, clsname)
        if not os.path.exists(cls_path):
            os.mkdir(cls_path)
        # 存放图片
        cv2.imwrite(os.path.join(cls_path, cur_frame_name), cur_frame)

        cur_id += 1

        if cur_id > len(lines):
            break
    reader.release()


def extract_imgs_from_videos(video_folder,label_folder,target_path,encoding='gbk'):
    """
    提取指定文件夹下的所有视频图像，不支持嵌套

    约定：video的名字和对应的label一样

    :param video_folder:
    :param label_folder:
    :param target_path:
    :param encoding:
    :return:
    """
    video_paths = []
    label_paths =[]
    for vname in os.listdir(video_folder):
        if vname.endswith('.wmv') or vname.endswith('.avi'):
            lname = os.path.basename(vname).split('.')[0]
            lname = lname + '.txt'
            lfile = os.path.join(label_folder, lname)
            if not os.path.exists(lfile):
                print('Label file not exits: {} . Skipping ... '.format(lfile,))
                continue
            video_paths.append(os.path.join(video_folder, vname))
            label_paths.append(os.path.join(label_folder,lname))

    for video_path,label_path in zip(video_paths,label_paths):
        print('Extract imgs from video : {}'.format(video_path,))
        extract_imgs_from_vido(video_path,label_path,target_path,encoding=encoding)


def shear_imgs(folder,target_folder,dsize=(660,880)):
    """裁取原始大图的指定大小的中央部分"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for imgname in os.listdir(folder):
        if imgname.endswith('.jpg'):
            img_path = os.path.join(folder, imgname)
            img = cv2.imread(img_path)

            # calculate img size
            o_h,o_w,_ = img.shape
            d_h,d_w = dsize
            # 如果原始圖片大小不足660x880,縮放,否則裁剪
            if o_h<d_h or o_w<d_w:
                img2 = cv2.resize(img, dsize=(d_w,d_h))
            else:
                h_start = (o_h-d_h)//2
                w_start =(o_w-d_w)//2
                # sheared img
                img2 = img[h_start:-h_start, w_start:-w_start]
            print('after shape : ', img2.shape)
            cv2.imwrite(os.path.join(target_folder, imgname), img2)


if __name__ == '__main__':
    # 将图片裁剪为880x660
    folder = r'D:\warelee\datasets\train\xception\train\bgg'
    target_folder = r'D:\warelee\datasets\train\xception\train\bg'
    for subf in os.listdir(folder):
        src_folder = os.path.join(folder,subf)
        if os.path.isdir(src_folder):
            dst_folder =os.path.join(target_folder,subf)
            shear_imgs(src_folder,dst_folder,dsize=(660,880))

    # shear_imgs(r'D:\tmp\nfl', r'D:\tmp2\nfl')

    # 从单个视频中提取图片，并根据标准文件分门别类存放
    # video_path = r'D:\cur_work\ultrasound\video\20180202_102752_236.wmv'
    # label_path = r'D:\cur_work\ultrasound\video\20180202_102752_236.txt'
    # target_path = r'D:\cur_work\ultrasound\video\imgs'
    # extract_imgs_from_vido(video_path, label_path, target_path)

    # 提取指定文件夹下的所有视频图像
    # video_folder =r'D:\originalmedicalimgs\16'
    # label_folder =r'D:\originalmedicalimgs\16\label'
    # target_path = r'D:\originalmedicalimgs\16\imgs'
    # extract_imgs_from_videos(video_folder,label_folder,target_path)

    # 采样
    # src_folder = r'D:\originalmedicalimgs\16\imgs\fl'
    # dst_folder =r'D:\test_imgs\fl'
    # sampling(src_folder,dst_folder,sep=4)
