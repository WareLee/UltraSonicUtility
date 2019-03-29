# 模型的准确度评估，混淆矩阵，分类报告，画roc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import json


def acc_cm_clsrep(y_true, y_pred, log_path=''):
    # 准确度评估，混淆矩阵，分类报告
    accu = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    clsrep = classification_report(y_true, y_pred)
    if log_path != '':
        with open(log_path, 'w', encoding='utf-8') as f:
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


def plot_roc(y_true, y_pred, y_props):
    """为xception定制的画roc"""

    # 如果输入的是保存结果的文件路径
    if isinstance(y_true, str):
        with open(y_true) as f:
            y_true = json.loads(f.read())
        with open(y_pred) as f:
            y_pred = json.loads(f.read())
        with open(y_props) as f:
            y_props = json.loads(f.read())

    std = {'nac': 'ac', 'nfl': 'fl', 'nhc': 'hc'}
    props = {}
    labels = {}

    for v in std.values():
        props[v] = []
        labels[v] = []

    for i, label in enumerate(y_true):
        if label == 'bg':
            continue

        # 如果实际是正样本，被预测为正样本，prop不变
        # 如果实际是正样本，被预测为其它，prop=1-prop
        # 如果实际是负样本，被预测为负样本，prop = 1- prop
        # 如果实际是负样本，被预测为其它，prop不变
        if label in std.values():
            if y_pred[i] == label:
                props[label].append(y_props[i])
            else:
                props[label].append(1 - y_props[i])
            labels[label].append(label)
        else:
            if label != y_pred[i]:
                props[std[label]].append(y_props[i])
            else:
                props[std[label]].append(1 - y_props[i])
            labels[std[label]].append(label)

    for v in props.keys():
        fpr, tpr, thresholde = roc_curve(labels[v], props[v], pos_label=[v])
        print('--------------------------------')
        print(v)
        # print(fpr)
        # print(tpr)
        thresholde = [int(float(hold) * 10000) / 10000 for hold in thresholde]
        print(thresholde)
        plt.plot(fpr, tpr, marker='o')
        plt.show()


if __name__ == '__main__':
    from xcep.xcep_app import test_on_img_folder
    y_true, y_pred, y_props = test_on_img_folder(r'D:\cls_images\sheared\test', r'D:\cls_images\sheared\error')
    acc_cm_clsrep(y_true,y_pred)
    plot_roc(y_true,y_pred,y_props)