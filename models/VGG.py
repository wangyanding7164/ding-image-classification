import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,features,class_num=1000,init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.dropout(p=0.5),
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,class_num),
        )
        if init_weights:
            self.initialize_weights()
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_features(cfg: list):
    layers=[]
    in_channels=3
    for v in cfg:
        if v =='M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        else:
            connv2d = nn.Conv2d(in_channels,out_channels=v,kernel_size=3,padding=1)
            layers += [connv2d, nn.ReLU(True)]
            in_channels=v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11' : [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13' : [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16' : [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19' : [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model