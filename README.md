# Pytorch-BaseClassification

# case1
conv(3,64)->avgpool->conv(64,128)->....->FC->softmax(-1)

classes_20.pth acc:0.7657

classes_80.pth acc:0.7514

classes_0.pth acc:0.6980

classes_40.pth acc:0.7573

classes_60.pth acc:0.7593

# case2
conv(3,64)->maxpool->conv(64,128)->...->FC->SoftMax(-1)
