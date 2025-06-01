import logging
from abc import ABC, abstractmethod
from typing import Any, Callable


class BaseFactory(ABC):
    """
    An abstract base class that defines the interface for a object factory.
    Subclasses must implement the `create` method.
    """

    # Internal registry for available objects
    registry: dict[str, Callable] = {}

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

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Method responsible for registering a new object in the factory, so that it can be created later.

        Args:
            name (str): The name of the object to register.

        Returns:
            A Callable.
        """

        def wrapper(wrapped_class: Callable) -> Callable:
            factory_name = cls.__name__.strip("Factory")

            if name in cls.registry:
                msg = f"{factory_name} {name} already exists. Will replace it"
                logging.warning(msg)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def get_object(cls, name: str, **kwargs) -> Any:
        if name not in cls.registry:
            logging.warning(
                f"{name} does not exist in the registry. Probably weren't implemented."
            )
            return None
        return cls.registry[name](**kwargs)
