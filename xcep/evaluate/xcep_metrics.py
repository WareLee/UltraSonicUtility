# 模型的准确度评估，混淆矩阵，分类报告，画roc
from sklearn.metrics import accuracy_score,auc, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import json
import numpy as np


def acc_cm_clsrep(y_true, y_pred, log_path='',abnormal=False):
    # 异常模式下，视非标准到非标准的预测为正确
    if abnormal:
        nstd = {'nac', 'nhc', 'nfl', 'afl','bg'}
        for i in range(len(y_true)):
            if y_true[i] in nstd:
                y_true[i]='nstd'
            if y_pred[i] in nstd:
                y_pred[i]='nstd'


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


def plot_roc_xcep(y_true, y_pred, y_props):
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
        print('auc : ',auc(fpr,tpr))
        print(fpr)
        print(tpr)
        thresholde = [int(float(hold) * 10000) / 10000 for hold in thresholde]
        print(thresholde)
        plt.plot(fpr, tpr, marker='o')
        plt.show()

# 通用版本
def plot_roc(y_label, y_score):

    # used for roc curve
    # y_score = [[], [], []]
    # y_label = [[], [], []]
    plt.figure()

    lw = 2

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    label_names = ['HC', 'AC', 'FL']
    for label, score, color, name in zip(y_label, y_score, colors, label_names):
        fpr, tpr, thresholds = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)

        # threshold
        right_index = (tpr + (1 - fpr) - 1)

        idx = np.argmax(right_index)
        tpr_val = tpr[idx]
        fpr_val = fpr[idx]
        thresh = thresholds[idx]

        plt.plot(fpr, tpr, color=color, lw=lw, label=name + '(auc: {:.2f})'.format(roc_auc))
        plt.plot(fpr_val, tpr_val, 'ro')
        label = '{:.2f}: ({:.2f}, {:.2f})'.format(thresh, fpr_val, tpr_val)
        plt.text(fpr_val, tpr_val, label)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')

    plt.show()

if __name__ == '__main__':
    import os
    import json
    from xcep.xcep_app import test_on_img_folder
    img_folder = r'D:\warelee\datasets\test\xception\hd_test'
    err_parent= 'nfl_epo4_hd_test'
    classes = ['hc', 'ac', 'fl', 'nhc', 'nac', 'nfl', 'bg']
    y_true, y_pred, y_props = test_on_img_folder(img_folder, os.path.join(r'D:\warelee\datasets\test\xception\tmp',err_parent),classes=classes)
    acc_cm_clsrep(y_true,y_pred,log_path=os.path.join(r'D:\warelee\datasets\test\xception\tmp',err_parent,err_parent+'.log'))
    with open(os.path.join(r'D:\warelee\datasets\test\xception\tmp',err_parent,'y_true.txt'),'w') as f:
        f.write(json.dumps(y_true))
    with open(os.path.join(r'D:\warelee\datasets\test\xception\tmp',err_parent,'y_pred.txt'),'w') as f:
        f.write(json.dumps(y_pred))
    with open(os.path.join(r'D:\warelee\datasets\test\xception\tmp',err_parent,'y_props.txt'),'w') as f:
        f.write(json.dumps(y_props))

    # parent_p = r'D:\warelee\datasets\test\xception\tmp\nfl_epo4_test'
    # with open(os.path.join(parent_p,'y_true.txt')) as f:
    #     y_true = json.loads(f.read())
    # with open(os.path.join(parent_p,'y_pred.txt')) as f:
    #     y_pred = json.loads(f.read())
    # acc_cm_clsrep(y_true,y_pred,log_path=os.path.join(parent_p,'nfl_epo4_test_ab.log'),abnormal=True)

    # 畫ROC曲綫
    # y_true='../../tmp/y_true.txt'
    # y_pred ='../../tmp/y_pred.txt'
    # y_props='../../tmp/y_props.txt'
    # plot_roc_xcep(y_true,y_pred,y_props)