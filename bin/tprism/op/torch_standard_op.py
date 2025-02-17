import numpy as np
import torch
from tprism.op.base import BaseOperator
from typing import List

class Reindex(BaseOperator):
    def __init__(self, parameters: List[str]):
        index = parameters[0].strip("[]").split(",")
        self.out = index
        pass

    def call(self, x):
        return x

    def get_output_template(self, input_template):
        if len(input_template) > 0 and input_template[0] == "b":
            return ["b"] + self.out
        else:
            return self.out


class Sigmoid(BaseOperator):
    def __init__(self, parameters: List[str]) -> None:
        pass

    def call(self, x):
        return torch.sigmoid(x)

    def get_output_template(self, input_template: List[str]) -> List[str]:
        return input_template


class Relu(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.relu(x)

    def get_output_template(self, input_template):
        return input_template


class Softmax(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.softmax(x,dim=-1)

    def get_output_template(self, input_template):
        return input_template

class Min1(BaseOperator):
    def __init__(self, parameters: List[str]):
        pass

    def call(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def get_output_template(self, input_template):
        return input_template
