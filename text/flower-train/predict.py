import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.AlexNet import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize(224,224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), f"{img_path} does not exist"
    img = Image.open(img_path)
    img = data_transforms(img)

    plt.imshow(img)
    #[N,C,H,W]
    img = data_transforms(img)

    img=torch.unsqueeze(img,dim=0)

    json_path = "../class_indices.json"
    assert os.path.exists(json_path), f"{json_path} does not exist"
    with open(json_path,"r") as f:
        class_indices = json.load(f)

    model = AlexNet(len(class_indices))

    weights_path = "../AlexNet.pth"
    assert os.path.exists(weights_path), f"{weights_path} does not exist"
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
