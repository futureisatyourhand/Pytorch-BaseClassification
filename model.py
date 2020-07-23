import torch.nn as nn
class Classification(nn.Module):
    #input size:128*128;
    def __init__(self,num_classes,train=True):
        super(Classification,self).__init__()
        self.num_classes=num_classes
        self.train=train
        self.convs=nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3,2),

            nn.Conv2d(64,128,3,1,1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),

            nn.Conv2d(128,256,3,2,1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,3,1,1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,512,3,1,1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512,728,3,2,1,bias=True),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),

        )
        self.fc=nn.Sequential(
            nn.Linear(11648,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,2),
           
        )
        self.classes= nn.Softmax(dim=-1)
        self.loss=nn.CrossEntropyLoss(reduce=True)
    def forward(self,x,targets):
        x=self.convs(x)
        x=x.view(x.shape[0],-1)
        print(x.shape)
        x=self.fc(x)
        print(targets.shape)
        if self.train:
            loss=self.loss(x,targets)
            return loss
        else:
            return self.classes(x)
