import torch

model_path = r'C:\Users\filos\OneDrive\Desktop\ETH\model-fusion\saved_models\cifar10_vgg11_bs32.t7'
model_path2 = r'C:\Users\filos\OneDrive\Desktop\ETH\model-fusion\saved_models\cifar10_resnet18_bs256.t7'
model1 = torch.load(model_path)
model2 = torch.load(model_path2)

print('Model 1')
for key in model1.keys():
    print(key,model1[key].shape)

print('Model 2')
for key in model2.keys():
    print(key,model2[key].shape)




