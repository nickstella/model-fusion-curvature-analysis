import enum

from torch import nn
from torchvision.models import vgg16, resnet18


class ModelType(enum.Enum):
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'

    def get_model(self, *args, **kwargs) -> nn.Module:
        if self == ModelType.RESNET18:
            num_channels = kwargs.pop('num_channels', 3)
            model = resnet18(*args, **kwargs)

            # The default channel size for resnet18 is 3, but we want to be able to change it
            if num_channels != 3:
                model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return model
        if self == ModelType.VGG16:
            return vgg16(*args, **kwargs)
        raise ValueError(f'Unknown architecture: {self}')
