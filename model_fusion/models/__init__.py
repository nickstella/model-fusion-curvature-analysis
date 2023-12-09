import enum

from torchvision.models import vgg16, resnet18

from model_fusion.models.lightning import BaseModel

class ModelType(enum.Enum):
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'

    def get_model(self, *args, **kwargs) -> BaseModel:
        if self == ModelType.RESNET18:
            return vgg16(*args, **kwargs)
        if self == ModelType.VGG16:
            return resnet18(*args, **kwargs)
        raise ValueError(f'Unknown architecture: {self}')