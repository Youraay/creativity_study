from abc import ABC, abstractmethod
from custom_types import Argument,Argument2, Noise, Evaluation
from typing import Generic, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..models.manager import ModelManager

class Evaluator(Generic[Argument] , ABC):
    @abstractmethod
    def evaluate(self, noise: Argument, *args, **kwargs):

        raise NotImplementedError("Method is not implementet yet")