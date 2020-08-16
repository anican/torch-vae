from torch import nn
from abc import abstractmethod

from .types import *


class BaseVAE(nn.Module):
    """
    BaseVAE module serves as a template for all variational autoencoder model
    subclasses.
    """

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, inputs: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, inputs: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

