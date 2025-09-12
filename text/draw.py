import numpy as np
import torch
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'Noto Sans CJK SC']  # 字体
plt.rcParams['axes.unicode_minus'] = False
from models.AlexNet import AlexNet
import itertools
class ConfusionMatrix(object):

    def __init__(self,num_classes: int,labels: list, normalize: bool, batch_size: int):
        self.matrix = np.zeros((num_classes,num_classes)) #全零二阶矩阵
        self.num_classes=num_classes
        self.labels=labels
        self.normalize=normalize
        self.batch_size=batch_size
    def update(self,preds,labels):
        for t,p in zip(preds,labels):
            self.matrix[t,p]+=1

    def summary(self):
        print(self.matrix)

    def plot_confusion(self):
        matrix=self.matrix
        classes=self.labels
        normalize=self.normalize
        title = 'Confusion matrix'
        cmap = plt.cm.Blues #颜色

        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks,classes,rotation=45, ha='right')
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






#数据预处理
data_transform=transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#获取图片路径
val_path=os.path.join(r"C:\Users\86188\Desktop\ding-image-classification\Data\data")
image_path=os.path.join(val_path,'val')
assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

val_dataset=datasets.ImageFolder(root=image_path,transform=data_transform)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=0)


num_classes=len(val_dataset.classes)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#加载模型
net=AlexNet(num_classes=num_classes)
model_pth_path=r"C:\Users\86188\Desktop\ding-image-classification\text\flower-train\AlexNet.pth"
assert os.path.exists(model_pth_path), "cannot find {} file".format(model_pth_path)
ckpt = torch.load(model_pth_path, map_location=device)
net.load_state_dict(ckpt, strict=False)
net.to(device)

#读取标签

json_label_path = r"C:\Users\86188\Desktop\ding-image-classification\text\flower-train\class_indices.json"  #存放标签的json文件
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
with open(json_label_path,'r') as f:
    labels = json.load(f)

labels= [label for _,label in labels.items()]


Confusion=ConfusionMatrix(len(labels),labels,normalize=True,batch_size=32)

net.eval()
with torch.no_grad():
    for val in tqdm(val_loader):
        val_img,val_labels=val
        outputs = net(val_img.to(device))
        outputs = torch.softmax(outputs,dim=1) #算每个类别的概率
        outputs = torch.argmax(outputs,dim=1) #得到最大概率的索引
        Confusion.update(outputs.to("cpu").numpy(),val_labels.to("cpu").numpy())
Confusion.plot_confusion()


