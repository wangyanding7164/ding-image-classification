import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from models.AlexNet import AlexNet


def main():
    #看是cpu还是gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    #数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),   #随机裁剪，将其裁剪成224
                                     transforms.RandomHorizontalFlip(),   #随即反转
                                     transforms.ToTensor(),               #转化成张量
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), #标准化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  #获取当前目录后退两个的目录地址
    image_path = os.path.join(data_root, "Data", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  #存在与否
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    train_nums=len(train_dataset)
    flower_list=flower_list = train_dataset.class_to_idx

    print("flower_list: ", flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),transform=data_transform["val"])
    val_num=len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    net=AlexNet(num_classes=len(flower_list),init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss() #损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.0002) #优化器

    save_path = './AlexNet.pth'
    best_acc= 0.0
    epochs=1
    train_steps=len(train_loader)

    for epoch in range(epochs):
        #train
        net.train()  #开启Dropout
        running_loss = 0.0  #统计平均损失
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()  #梯度清零
            outputs = net(images.to(device)) #前向传递
            loss=loss_function(outputs,labels.to(device))  #计算损失
            loss.backward() #后向传递
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        net.eval()
        acc=0.0
        with torch.no_grad():
            for data_test in val_loader:
                test_images, test_labels = data_test
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)


    print('Finished Training')

main()