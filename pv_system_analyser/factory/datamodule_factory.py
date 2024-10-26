from typing import Any, Literal

from .interface import BaseFactory


class DataModuleFactory(BaseFactory):
    @classmethod
    def create(
        cls, name: Literal["classification", "segmentation"], **kwargs
    ) -> Any:
        return cls.get_object(name, **kwargs)
        