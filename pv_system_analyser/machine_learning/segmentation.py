from torchvision.models._api import WeightsEnum

def preprocess_data(weights: WeightsEnum, images: list):
    transforms = weights.transforms()
    return [transforms(image) for image in images]


