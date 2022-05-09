import torch.nn as nn
from torchvision.models import vgg11_bn

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg11_base,self).__init__()
        self.vggmodel=vgg11_bn(pretrained=False).features[:-1]
        self.vggmodel[0]=nn.Conv2d(1,64,kernel_size = 3, padding= 1)
        # self.vggmodel.add_module('our_1',nn.Conv2d(512, 4096, kernel_size=3, stride=1, padding=1))
        # self.vggmodel.add_module('our_2', nn.BatchNorm2d(4096, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True))
        # self.vggmodel.add_module('our_3', nn.ReLU())

    def forward(self, x):
        x = self.vggmodel(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG,self).__init__()
        self.backbone=vgg16_base()
        self.avgpool = nn.AvgPool2d(kernel_size=(1,9), stride=(1,1))
        self.maxpool = nn.MaxPool1d(kernel_size=45, stride=1, padding=0)
        self.linear = nn.Linear(in_features=512, out_features=num_classes)
        self.activate = nn.Softmax(dim=1)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, input, label = None):
        result = self.backbone(input)
        # import pdb;pdb.set_trace()
        #result = self.avgpool(result)
        result = result.view(result.size(0), result.size(1), -1)
        result = self.maxpool(result)
        result = result.reshape(result.size(0), -1)
        result = self.linear(result)
        
        if label != None:
            _, label = label.max(-1)
            pred = self.activate(result)
            loss = self.criteria(result, label)
        # import pdb; pdb.set_trace()
            return loss, result, pred, label
        else:
            pred = self.activate(result)
            _, pred_label = pred.max(-1)
            return pred_label
