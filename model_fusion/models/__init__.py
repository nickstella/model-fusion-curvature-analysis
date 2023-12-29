import enum

from torch import nn
from torchvision.models import vgg11, vgg11_bn, resnet18


class ModelType(enum.Enum):
    RESNET18 = 'resnet18'
    VGG11 = 'vgg11'

    def remove_bias(self, model: nn.Module):
        model = model.apply(lambda m: m.register_parameter('bias', None))
        return model

    def get_model(self, *args, **kwargs) -> nn.Module:
        bias = kwargs.pop('bias', False)
        num_channels = kwargs.pop('num_channels', 3)
        batch_norm = kwargs.pop('batch_norm', False)

        model = None

        if self == ModelType.RESNET18:
            if not batch_norm:
                kwargs['norm_layer'] = lambda x: nn.Identity()

            model = resnet18(*args, **kwargs)

            # The default channel size for resnet18 is 3, but we want to be able to change it
            if num_channels != 3:
                model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self == ModelType.VGG11:
            model = vgg11(*args, **kwargs) if not batch_norm else vgg11_bn(*args, **kwargs)

            if num_channels != 3:
                model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)

        if model is None:
            raise ValueError(f'Unknown architecture: {self}')

        if not bias:
            model = self.remove_bias(model)

        return model


if __name__ == '__main__':
    model = ModelType.RESNET18.get_model(num_classes=10, bias=False)
    for index, (name, param) in enumerate(model.named_parameters()):
        print(index, name, param.shape)
