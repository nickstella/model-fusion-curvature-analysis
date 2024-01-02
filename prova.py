import torch

model_path = r'C:\Users\filos\OneDrive\Desktop\ETH\loss-landscape\sidak\cifar10_resnet18_bs32.t7'
model_path2 = r'C:\Users\filos\OneDrive\Desktop\ETH\loss-landscape\sidak\cifar10_resnet18img_bs32.t7'
model1 = torch.load(model_path)
model2 = torch.load(model_path2)

for key in model1.keys():
    print(key,model1[key].shape)


