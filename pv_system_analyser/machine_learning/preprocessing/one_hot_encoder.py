from functools import wraps
from typing import Callable

import torch


def validate_one_hot_input(fn: Callable) -> Callable:
    """
    Validates that the input label is a string.
    This decorator it's used to wrap the "__call__" method of the OneHotEncoder class.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function that validates the input label.
    """

    @wraps(fn)
    def wrapper(self, label: str) -> torch.Tensor:
        if not isinstance(label, str):
            raise TypeError("Input must be a string")
        return fn(self, label)

    return wrapper


class ToOneHot:
    """
    Applies one-hot encoding to input labels.

    Any string label is converted to one-hot encoded tensor.
    It maintains a list of unique labels encountered, and uses this to look up the one-hot encoding for a given label.

    Parameters:
        n_classes (int): Number of unique classes for one-hot encoding.

    Returns:
        torch.Tensor: One-hot encoded tensor representation of the input label.

    Raises:
        TypeError: If input label is not a string.

    Examples:
        >>> encoder = ToOneHot(n_classes=3)
        >>> encoder("label_1")  # Returns: tensor([1., 0., 0.])
    """

    def __init__(self, n_classes: int) -> None:
        self.one_hot_labels = torch.eye(n_classes)
        self.labels: list[str] = []

    @validate_one_hot_input
    def __call__(self, label: str) -> torch.Tensor:
        if label not in self.labels:
            self.labels.append(label)
        return self.one_hot_labels[self.labels.index(label)]
