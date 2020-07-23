# Pytorch-BaseClassification

# The influence of pooling for classification task
# case1
conv(3,64)->avgpool->conv(64,128)->....->FC->softmax(-1)

classes_20.pth acc:0.7657

classes_80.pth acc:0.7514

classes_0.pth acc:0.6980

classes_40.pth acc:0.7573

classes_60.pth acc:0.7593

# case2
conv(3,64)->maxpool->conv(64,128)->...->FC->SoftMax(-1)

classes_20.pth acc:0.7909

classes_80.pth acc:0.7593

classes_0.pth acc:0.7054

classes_40.pth acc:0.7756

classes_60.pth acc:0.7776

# case3:only 
According to the experiment of case2, maximum pooling is slightly better than average pooling, and we only use maximum pooling behind the network.

conv->conv->conv->--——>maxpool2d->conv->maxpool2d-fc->softmax(-1)

classes_20.pth acc:0.8161

classes_80.pth acc:0.8028

classes_0.pth acc:0.7143

classes_40.pth acc:0.7998

classes_60.pth acc:0.8265
# case4:global average pooling 
conv->conv->conv->conv--->global average pooling ->fc->softmax(-1)

classes_20.pth acc:0.8067

classes_80.pth acc:0.7899

classes_0.pth acc:0.6975

classes_40.pth acc:0.8077

