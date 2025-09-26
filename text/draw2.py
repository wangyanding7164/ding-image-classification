import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools

# 导入模型
from AlexNet import AlexNet

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'Noto Sans CJK SC']  # 字体
plt.rcParams['axes.unicode_minus'] = False
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list, normalize: bool, batch_size: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.normalize = normalize
        self.batch_size = batch_size

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):

        # print(matrix)
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize
        title = 'Confusion matrix'
        cmap = plt.cm.Blues  # 绘制的颜色

        print('normalize: ', normalize)

        """
         - matrix : 计算出的混淆矩阵的值
         - classes : 混淆矩阵中每一行每一列对应的列
         - normalize : True:显示百分比, False:显示个数
         """
        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            print("显示百分比：")
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(matrix)
        else:
            print('显示具体数字：')
            print(matrix)
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else '.0f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('my_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    # ------------需要修改的地方--------
    num_classes = 10
    normalize = True  # normalize：True-百分比; False-个数
    batch_size = 32
    # --------------------------------

    # 注意：图像预处理需要与模型训练时一致
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_path = os.path.join(r"")  # 数据集的地址
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = AlexNet(num_classes=num_classes)
    # 加载模型文件
    model_weight_path = "" #模型的地址
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    ckpt = torch.load(model_weight_path, map_location=device)
    # ----------如果报类似"conv_head.weight"的错，添加下列pop语句-----------
    # model = create_model(num_classes=num_classes).to(device)
    # ckpt = model.load_wieght(model_weight_path)
    # ckpt.pop("conv_head.weight")
    # -----------------------------------------------------------------
    net.load_state_dict(ckpt, strict=False)
    net.to(device)

    # 读取 class_indict
    json_label_path = 'flower-train/class_indices.json'  #写标签列表
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels, normalize=normalize, batch_size=batch_size)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1) #算每个类别的概率
            outputs = torch.argmax(outputs, dim=1) #获取最大值的索引
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()