"""
前提要求：图片库的目录结构中annotations.csv文件必须与相对应的图片在同一目录下
"""
import os
import json
import shutil


def getAllFolders(root_dirs, excepted_dirs=[]):
    """
    获取根目录及其所有后代目录
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :return: list[str]
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    # 全部转换为绝对路径
    root_dirs = [os.path.abspath(it) for it in root_dirs]
    excepted_dirs = [os.path.abspath(it) for it in excepted_dirs]

    # 获取所有目录
    src_folders = root_dirs.copy()
    i = 0
    while (i < len(src_folders)):
        # 第i个目录下的子文件夹
        folder_i = src_folders[i]
        for it in os.listdir(folder_i):
            tt = os.path.join(folder_i, it)
            if os.path.exists(tt) and os.path.isdir(tt) and (tt not in excepted_dirs):
                src_folders.append(tt)
        i += 1

    return src_folders


def getAllCsv(root_dirs, excepted_dirs=[], csv_name='annotations.csv'):
    """
    获取指定目录下及其子孙目录下的所有annotations.csv文件
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :param csv_name:
    :return:
    """
    csvs = []
    src_folders = getAllFolders(root_dirs, excepted_dirs)
    for folder in src_folders:
        tmp = os.path.join(folder, csv_name)
        if os.path.exists(tmp):
            csvs.append(tmp)
    return csvs


def updateAllCsv(root_dirs, excepted_dirs=[], csv_name='annotations.csv', delete_json=None):
    """
    更新csv文件，去掉image不存在的记录，去掉delete指定的json文件中有的.删除没有bbox的记录，并规范化类别名称：去掉切面二字
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :param csv_name:
    :param delete_json: 指定的json文件中包含要去除的图片信息
    :return:
    """
    del_records = []
    if delete_json != None:
        if delete_json.endswith('.json'):
            with open(delete_json, encoding='utf-8') as f:
                json_str = f.read()
            del_records = json.loads(json_str).keys()
        elif delete_json.endswith('.csv'):
            with open(delete_json, encoding='utf-8') as f:
                json_str = f.readlines()
            del_records = [line.strip().split(',')[0] for line in json_str]
        else:
            print('Warning : Could not parse the delete_file : ', delete_json)

    csvs = getAllCsv(root_dirs, excepted_dirs, csv_name=csv_name)
    for csv in csvs:
        lines = []
        with open(csv, encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                items = line.split(',')
                if items[-1].strip() == '' or items[1].strip() == '' or int(items[1]) < 0 or int(items[2]) < 0 or int(
                        items[3]) < 0:
                    continue
                if (not os.path.exists(os.path.join(os.path.split(csv)[0], items[0]))) or (items[0] in del_records):
                    continue
                lines.append(line.replace('切面\n', '\n'))
        with open(csv, 'w', encoding='utf-8') as f:
            f.writelines(lines)


def statisticCls(root_dirs, excepted_dirs=[], csv_name='annotations.csv'):
    """
    统计指定目录下所有记录中各个类别的数量
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :param csv_name:
    :return:
    """
    result = {}
    csvs = getAllCsv(root_dirs, excepted_dirs, csv_name)

    for csv in csvs:
        with open(csv, encoding='utf-8') as f:
            for line in f.readlines():
                cls = line.strip().split(',')[-1]
                if cls in result.keys():
                    result[cls] += 1
                else:
                    result[cls] = 1

    return result


def mergeFilesAndFolders2(root_dirs, dest_dir, excepted_dirs=[], csv_name='annotations.csv', rvduplication=True,
                          sep_cls_folder=False, **howmany):
    """
    合并指定目录下的图片和csv到目的文件夹，自动过滤掉包含了目的路径dest_dir的root_dir. 同时也会自动删除重复的图片
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param dest_dir: str, 存放合并后数据的目的目录
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :param csv_name:
    :param howmany: 指定各类别数据要合成多少,没指定的类别不合成，合成完制定量或者已经没有更多数据的时候停止合成，如果不提供，就合并所有数据。ex: 丘脑标准=2300
    :param rvduplication: 是否去重
    :return:
    """
    # 如果目的文件夹不存在
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    # 清空目的文件夹
    for it in os.listdir(dest_dir):
        tm = os.path.join(dest_dir, it)
        if os.path.isdir(tm):
            shutil.rmtree(tm)
        elif os.path.isfile(tm):
            os.remove(tm)
        else:
            print('Maybe you can delete useless link or something manually. : ', tm)

    # 记录合并的各类别的数量
    if len(howmany.keys()) > 0:
        record = {}
        for key in howmany.keys():
            record[key] = 0

    csvs = getAllCsv(root_dirs, excepted_dirs, csv_name)
    # 对含有autocsv的文件放到最后
    for csv in csvs.copy():
        if csv.find('autocsv') >= 0:
            csvs.remove(csv)
            csvs.append(csv)

    # 如果源图片目录中有目的地目录，要去除
    for it in csvs.copy():
        if os.path.abspath(it).find(dest_dir) >= 0:
            csvs.remove(it)

    imgs_set = set()

    # 如果目标文件的annotations.csv已存在，删除
    if os.path.exists(os.path.join(dest_dir, 'annotations.csv')):
        os.remove(os.path.join(dest_dir, 'annotations.csv'))

    print('Merging files and folders ....')
    for csv in csvs:
        lines = []
        parent_dir = os.path.split(csv)[0]
        with open(csv, encoding='utf-8') as f:
            for line in f.readlines():
                if not line.endswith('\n'):
                    line += '\n'
                if rvduplication:
                    items = line.split(',')
                    if items[0] not in imgs_set:

                        # 根据指定的数量合成
                        if len(howmany.keys()) > 0:
                            if items[-1].strip() in howmany.keys():
                                if record[items[-1].strip()] < howmany[items[-1].strip()]:
                                    record[items[-1].strip()] += 1
                                    # todo shan, 只保存2000张之后的图片
                                    if record[items[-1].strip()] < 2000:
                                        continue
                                else:
                                    continue
                            else:
                                continue
                        # 增加文件记录
                        imgs_set.add(items[0])
                        lines.append(line)

                        # 是否根据不同类别单独存放图片
                        if sep_cls_folder:
                            cls_path = os.path.join(dest_dir, items[-1].strip())
                            if not os.path.exists(cls_path):
                                os.mkdir(cls_path)
                        else:
                            cls_path = dest_dir

                        # 复制图片
                        if not os.path.exists(os.path.join(parent_dir, items[0])):
                            continue
                        shutil.copy(os.path.join(parent_dir, items[0]), os.path.join(cls_path, items[0]))
                    else:
                        print('Image already exists: ', items[0])
                        # print('Duplication image : ',items[0])
                        # print('Deleting Dup image : ',os.path.join(parent_dir,items[0]))
                        # if os.path.exists(os.path.join(parent_dir,items[0])):
                        #     os.remove(os.path.join(parent_dir,items[0]))

                else:
                    # 根据指定的数量合成
                    if len(howmany.keys()) > 0:
                        if items[-1].strip() in howmany.keys():
                            if record[items[-1].strip()] < howmany[items[-1].strip()]:
                                record[items[-1].strip()] += 1
                            else:
                                continue
                        else:
                            continue

                    lines.append(line)
                    shutil.copy(os.path.join(parent_dir, items[0]), os.path.join(dest_dir, items[0]))

        with open(os.path.join(dest_dir, 'annotations.csv'), 'a+', encoding='utf-8') as f:
            f.writelines(lines)

    # 自动生成合成后的数据统计信息到statistics.txt
    if len(howmany.keys()) == 0:
        record = statisticCls([dest_dir])
    with open(os.path.join(dest_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(str(record))


def mergeFilesAndFolders(root_dirs, dest_dir, excepted_dirs=[], csv_name='annotations.csv', rvduplication=True,
                         **howmany):
    """
    合并指定目录下的图片和csv到目的文件夹，自动过滤掉包含了目的路径dest_dir的root_dir. 同时也会自动删除重复的图片
    :param root_dirs: str or list[str]， 指定的根目录（们）
    :param dest_dir: str, 存放合并后数据的目的目录
    :param excepted_dirs: list[str], 要去除的例外目录及其后代目录
    :param csv_name:
    :param howmany: 指定各类别数据要合成多少,没指定的类别不合成，合成完制定量或者已经没有更多数据的时候停止合成，如果不提供，就合并所有数据。ex: 丘脑标准=2300
    :param rvduplication: 是否去重
    :return:
    """
    # 如果目的文件夹不存在
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    # 清空目的文件夹
    for it in os.listdir(dest_dir):
        tm = os.path.join(dest_dir, it)
        if os.path.isdir(tm):
            shutil.rmtree(tm)
        elif os.path.isfile(tm):
            os.remove(tm)
        else:
            print('Maybe you can delete useless link or something manually. : ', tm)

    # 记录合并的各类别的数量
    if len(howmany.keys()) > 0:
        record = {}
        for key in howmany.keys():
            record[key] = 0

    csvs = getAllCsv(root_dirs, excepted_dirs, csv_name)
    # 对含有autocsv的文件放到最后
    for csv in csvs.copy():
        if csv.find('autocsv') >= 0:
            csvs.remove(csv)
            csvs.append(csv)

    # 如果源图片目录中有目的地目录，要去除
    for it in csvs.copy():
        if os.path.abspath(it).find(dest_dir) >= 0:
            csvs.remove(it)

    imgs_set = set()

    # 如果目标文件的annotations.csv已存在，删除
    if os.path.exists(os.path.join(dest_dir, 'annotations.csv')):
        os.remove(os.path.join(dest_dir, 'annotations.csv'))

    print('Merging files and folders ....')
    for csv in csvs:
        lines = []
        parent_dir = os.path.split(csv)[0]
        with open(csv, encoding='utf-8') as f:
            for line in f.readlines():
                if not line.endswith('\n'):
                    line += '\n'
                if rvduplication:
                    items = line.split(',')
                    if items[0] not in imgs_set:

                        # 根据指定的数量合成
                        if len(howmany.keys()) > 0:
                            if items[-1].strip() in howmany.keys():
                                if record[items[-1].strip()] < howmany[items[-1].strip()]:
                                    record[items[-1].strip()] += 1
                                else:
                                    continue
                            else:
                                continue
                        # 增加文件记录
                        imgs_set.add(items[0])
                        lines.append(line)
                        # 复制图片
                        shutil.copy(os.path.join(parent_dir, items[0]), os.path.join(dest_dir, items[0]))
                    else:
                        print('Image already exists: ', items[0])
                        # print('Duplication image : ',items[0])
                        # print('Deleting Dup image : ',os.path.join(parent_dir,items[0]))
                        # if os.path.exists(os.path.join(parent_dir,items[0])):
                        #     os.remove(os.path.join(parent_dir,items[0]))

                else:
                    # 根据指定的数量合成
                    if len(howmany.keys()) > 0:
                        if items[-1].strip() in howmany.keys():
                            if record[items[-1].strip()] < howmany[items[-1].strip()]:
                                record[items[-1].strip()] += 1
                            else:
                                continue
                        else:
                            continue

                    lines.append(line)
                    shutil.copy(os.path.join(parent_dir, items[0]), os.path.join(dest_dir, items[0]))

        with open(os.path.join(dest_dir, 'annotations.csv'), 'a+', encoding='utf-8') as f:
            f.writelines(lines)

    # 自动生成合成后的数据统计信息到statistics.txt
    if len(howmany.keys()) == 0:
        record = statisticCls([dest_dir])
    with open(os.path.join(dest_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(str(record))


def delAndUpdateCsvOrFloder(root_dirs, excepted_dirs=[], need2delete=None):
    """删除不在记录中的图片，删除图片不存在的记录，删除重复出现的图片并更新csv"""
    # 删除csv中多余的记录
    updateAllCsv(root_dirs, excepted_dirs=excepted_dirs, delete_json=need2delete)
    # 删除文件中多余的图片
    delNotNotedImgs(root_dirs, excepted_dirs=excepted_dirs)
    # 对重复出现在多个文件夹中的图片，删除后来出现的
    csvs = getAllCsv(root_dirs, excepted_dirs=excepted_dirs)
    for csv in csvs.copy():
        if csv.find('autocsv') >= 0:
            csvs.remove(csv)
            csvs.append(csv)
    img_set = set()
    for csv in csvs:
        with open(csv, encoding='utf-8') as ff:
            for line in ff.readlines():
                items = line.strip().split(',')
                if items[0] in img_set:
                    img_p = os.path.join(os.path.split(csv)[0], items[0])
                    if os.path.exists(img_p):
                        print('Deleting Duplication images: ', img_p)
                        os.remove(img_p)
                else:
                    img_set.add(items[0])

    # 由于重复的图片被删除，所以会有大量多余的记录，故，再更新一次csv
    updateAllCsv(root_dirs, excepted_dirs=excepted_dirs)


def delNotNotedImgs(root_dirs, excepted_dirs=[]):
    """删除那些在csv文件中没有记录的图片"""
    for folder in getAllFolders(root_dirs, excepted_dirs=excepted_dirs):
        # 当前文件夹下的所有jpg
        all_jpg = []
        for jpg in os.listdir(folder):
            if jpg.endswith('.jpg'):
                all_jpg.append(jpg)
        # 当前目录下的cvs中记录的jpg
        csv_p = os.path.join(folder, 'annotations.csv')
        if os.path.exists(csv_p):
            with open(csv_p, encoding='utf-8') as ff:
                csv_imgs = [line.strip().split(',')[0] for line in ff.readlines()]

            # 取差，然后删除多余的图片
            dup_imgs = set(all_jpg) - set(csv_imgs)
            for dup in dup_imgs:
                tmp = os.path.join(folder, dup)
                print('Deleting not noted image: ', tmp)
                os.remove(tmp)
        else:
            # 删除当前floder下的所有图片
            for dup in all_jpg:
                tmp = os.path.join(folder, dup)
                print('Deleting not noted image: ', tmp)
                os.remove(tmp)


def correctCls(will_correcte_dirs, be_refered_dirs):
    """
    参照质检之后的文件，纠正原始数据的类别标签。 对质检之后的数据有目录结构的要求：必须是如： 腹部/标准 腹部/基本标准
    :param will_correcte_dirs: str, 将要被纠正的目录
    :param be_refered_dirs: list[str] or str , 用于参照的，质检过后的图片
    :return:
    """
    # 找到所有被参照的图片
    referred_floders = getAllFolders(be_refered_dirs)
    referred_imgs = []
    image_names = []
    image_labels = []
    for folder in referred_floders:
        for imgname in os.listdir(folder):
            if imgname.endswith('.jpg'):
                referred_imgs.append(os.path.join(folder, imgname))
    for imgp in referred_imgs:
        parent, name = os.path.split(imgp)
        parent2, small_cls = os.path.split(parent)
        _, big_cls = os.path.split(parent2)
        if small_cls in {'标准', '非标准'}:
            image_names.append(name)
            image_labels.append(big_cls + small_cls)

    # 找到所有的将被更新的csv文件，然后按图片名称更新
    will_csvs = getAllCsv(will_correcte_dirs)
    for csv in will_csvs:
        lines = []
        with open(csv, encoding='utf-8') as f:
            for line in f.readlines():
                items = line.strip().split(',')

                if items[0] in image_names:
                    index = image_names.index(items[0])
                    items[-1] = image_labels[index]
                    lines.append(','.join(items) + '\n')
                else:
                    lines.append(line)
        with open(csv, 'w', encoding='utf-8') as ff:
            ff.writelines(lines)


if __name__ == '__main__':
    root_dirs = r'D:\hnuMedical\ImageWare'
    excepted_dirs = [r'D:\hnuMedical\ImageWare\deleted', r'D:\hnuMedical\ImageWare\merged_images',
                     r'D:\hnuMedical\ImageWare\bg', r'D:\hnuMedical\ImageWare\D2', r'D:\hnuMedical\ImageWare\HD',
                     r'D:\hnuMedical\ImageWare\HD2']
    # delete_json = r'D:\hnuMedical\ImageWares\deleted\jibiao.csv'

    # 更新csv,去掉image不存在的记录，去掉bbox或者存在-1标识的记录,删除delete指定的记录
    # updateAllCsv(root_dirs,excepted_dirs=excepted_dirs,delete_json=r'D:\hnuMedical2\ImageWares\deleted\delete.csv')
    # 删除多余的图片&重复的图片并更新csv
    # delAndUpdateCsvOrFloder(root_dirs,excepted_dirs,need2delete=delete_json)

    # will_correcte_dirs=[r'D:\hnuMedical2\ImageWares\autocsv',r'D:\hnuMedical2\ImageWares\HD',r'D:\hnuMedical2\ImageWares\KL',r'D:\hnuMedical2\ImageWares\new_added']
    # be_refered_dirs=r'D:\hnuMedical2\ImageWares\errorimgs'
    # correctCls(will_correcte_dirs,be_refered_dirs)

    # excepted_dirs=[r'D:\hnuMedical2\ImageWares\deleted',r'D:\hnuMedical2\ImageWares\errorimgs']
    # need2delete=r'D:\hnuMedical2\ImageWares\deleted\delete.csv'
    # root_dirs =r'D:\hnuMedical2\ImageWares'
    # delAndUpdateCsvOrFloder(root_dirs,excepted_dirs=excepted_dirs,need2delete=need2delete)

    # root_dirs=[r'D:\workspaces\hnuMedical\ImageWare']
    # excepted_dirs =[r'D:\workspaces\hnuMedical\ImageWare\merged_images']
    # excepted_dirs=[]

    # test getAllFolders
    # result = getAllFolders(root_dirs,excepted_dirs)
    # print(len(result))
    # print(result)

    # test getAllCsv
    # result = getAllCsv(root_dirs,excepted_dirs)
    # print(len(result))
    # print(result)

    # test mergeFilesAndFolders 丘脑标准': 49, '腹部标准': 68
    mergeFilesAndFolders2(root_dirs, dest_dir=r'D:\tmp', excepted_dirs=excepted_dirs, sep_cls_folder=True, 丘脑标准=3000,
                          腹部标准=3000, 股骨标准=3000, 丘脑非标准=3000, 腹部非标准=3000, 股骨非标准=3000)
    # mergeFilesAndFolders(root_dirs,r'C:\Users\WareLee\Desktop\test\images_merged')

    # test updateAllCsv
    # roor_dirs = r'D:\hnuMedical2\ImageWares'
    # excepted_dirs = [r'D:\hnuMedical2\ImageWares\errorimgs']
    # delAndUpdateCsvOrFloder(roor_dirs,excepted_dirs=excepted_dirs,need2delete=r'D:\hnuMedical2\ImageWares\deleted\delete.csv')

    # rest statisticCls

    # result = statisticCls(root_dirs,excepted_dirs=excepted_dirs)
    # result = mergeFilesAndFolders(root_dirs, excepted_dirs=excepted_dirs)
    # print(result)
