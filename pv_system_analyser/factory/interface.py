from abc import ABC, abstractmethod
from typing import Any
import logging


class BaseFactory(ABC):
    """
    An abstract base class that defines the interface for a object factory. 
    Subclasses must implement the `create` and `register` methods.
    """
    
    # Internal registry for available objects
    registry = {}
    
    @abstractmethod
    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """
        Method responsible for creating a new instance of a object, given any necessary arguments.
        
        Args:
            name (str): The object's name to create.
            **kwargs: Any additional arguments required to create the object instance.
        
        Returns:
            An instance of the requested object.
        """        
        raise NotImplementedError()
    
    @abstractmethod
    @classmethod
    def register(self, name: str) -> Any:
        """
        Method responsible for registering a new object in the factory, so that it can be created later.
        
        Args:
            name (str): The name of the object to register.
        """        
        raise NotImplementedError()
