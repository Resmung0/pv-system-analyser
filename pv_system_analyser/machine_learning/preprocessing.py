import torch


class ToOneHot:
    def __init__(self, n_classes):
        self.one_hot_labels = torch.eye(n_classes)
        self.labels = []

    def __call__(self, label):
        if label not in self.labels:
            self.labels.append(label)
        return self.one_hot_labels[self.labels.index(label)]