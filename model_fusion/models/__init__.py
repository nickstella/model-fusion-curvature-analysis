import enum

from torch import nn
from torchvision.models import vgg11, vgg16, resnet18
from copy import deepcopy


class ModelType(enum.Enum):
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'
    VGG11 = 'vgg11'

    def remove_bias(self, model: nn.Module):
        model = model.apply(lambda m: m.register_parameter('bias', None))
        return model

    def replace_bn_with_identity(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                self.replace_bn_with_identity(child)



    def get_model(self, *args, **kwargs) -> nn.Module:
        bias = kwargs.pop('bias', False)
        num_channels = kwargs.pop('num_channels', 3)

        model = None

        if self == ModelType.RESNET18:
            model = resnet18(*args, **kwargs)

            # The default channel size for resnet18 is 3, but we want to be able to change it
            if num_channels != 3:
                model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self == ModelType.VGG16:
            model = vgg16(*args, **kwargs)

        if self == ModelType.VGG11:
            model = vgg11(*args, **kwargs)

            if num_channels != 3:
                model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)

        if model is None:
            raise ValueError(f'Unknown architecture: {self}')

        if not bias:
            model = self.remove_bias(model)

        # skip batch norm layers
        self.replace_bn_with_identity(model)

        return model


if __name__ == '__main__':
    model = ModelType.RESNET18.get_model(num_classes=10, bias=False)
    for index, (name, param) in enumerate(model.named_parameters()):
        print(index, name, param.shape)
