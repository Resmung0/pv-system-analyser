from typing import Callable, Literal
import logging

import torch
from lightning.pytorch import LightningModule

from .interface import BaseFactory


class ModelFactory(BaseFactory):
    """
    Factory class for creating models.
    """
    
    @classmethod
    def create(cls, name: str, **kwargs) -> None | LightningModule:
        """
        Factory command to create the model. 
        
        This method gets the appropriate LightningModule class from the registry
        and creates an instance of it, while passing in the parameters given in ``kwargs``.
        
        Args:
            name (str): The name of the model to create.
        
        Returns:
            An instance of the model that is created.
        """
        if name not in cls.registry:
            msg = f"Model {name} does not exist in the registry. Probably weren't implemented."
            logging.warning(msg)
            return None
        
        if 'mode' not in kwargs:
            kwargs['mode'] = 'cpu'
        device = cls._choose_process_device(kwargs['mode'])
        model = cls.registry[name](**kwargs)
        return model.to(device)

    @staticmethod
    def _choose_process_device(mode: Literal['training', 'inference']) -> str:
        """
        Chooses the appropriate device for processing based on the given mode.
        
        Args:
            mode (Literal['training', 'inference']): The mode of operation, either 'training' or 'inference'.
        
        Returns:
            str: The device to be used for processing, either 'cuda' or 'cpu'.
        """
        match mode:
            case 'trainig':
                if torch.cuda.is_available():
                    process_unity = 'cuda'
                else:
                    logging.warning('No Cuda GPU support found! Training model in CPU')
                    process_unity = 'cpu'
            case 'inference':
                process_unity = 'cpu'
        return torch.device(process_unity)
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Method to register a model to the internal registry.
        
        Args:
            name (str): The name of the executor.
    
        Returns:
            The Executor class itself.
        """
        def wrapper(wrapped_class: LightningModule) -> Callable:
            if name in cls.registry:
                msg = f"Model {name} already exists. Will replace it"
                logging.warning(msg)
            cls.registry[name] = wrapped_class
            return wrapped_class
        return wrapper
    