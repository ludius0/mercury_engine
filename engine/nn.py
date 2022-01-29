# scripts
from .func_base import Func, setattr_value

class ReLU(Func):
    """
    >>> (Value(-42).relu(), Value(42).relu())
    >>> (Value(0), Value(42))
    """
    @staticmethod
    def forward(ctx, x):
        ctx.saved_values.extend([x])
        return x if x > 0 else 0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_values
        return grad_output * (input >= 0)
setattr_value(ReLU)

class Neuron:
    def __init__(self):
        pass

    def __call__(self):
        pass