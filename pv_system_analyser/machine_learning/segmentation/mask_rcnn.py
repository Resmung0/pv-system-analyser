from typing import Literal
from torchvision.models.detection import MaskRCNN
from torchvision.models import get_model

def create_mask_rcnn(backbone: Literal['classic', 'alternative'] = 'classic') -> MaskRCNN:
    match backbone:
        case "classic":
            backbone_name = "maskrcnn_resnet50_fpn_v2"
        case "alternative":
            backbone_name = "maskrcnn_resnet50_fpn"
        case _:
            raise ValueError("This backbone weren't implemented.")
    model = get_model(backbone_name, weights='DEFAULT')
    return model
