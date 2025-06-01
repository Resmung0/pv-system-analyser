from typing import Literal
import logging

import torch
from lightning.pytorch import LightningModule

from .interface import BaseFactory


class ModelFactory(BaseFactory):
    """
    Factory class for creating models.
    """
    
    @classmethod
    def create(
        cls,
        name: str,
        mode: Literal['training', 'inference'] = 'inference',
        **kwargs
    ) -> LightningModule:
        """
        Factory command to create the model. 
        
        This method gets the appropriate LightningModule class from the registry
        and creates an instance of it, while passing in the parameters given in ``kwargs``.
        
        Args:
            name (str): The name of the model to create.
        
        Returns:
            An instance of the model that is created.
        """        
        device = cls._choose_process_device(mode)
        model = cls.get_object(name, **kwargs)
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
    