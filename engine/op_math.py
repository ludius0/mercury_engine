# libs
import math
# scripts
from .func_base import Func, setattr_value

class Add(Func):
    """
    >>> Value(1).add(1)
    >>> Value(2)
    """
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
setattr_value(Add)

class Mul(Func):
    """
    >>> Value(1).mul(2)
    >>> Value(2)
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.saved_values.extend([x, y])
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_values
        return y * grad_output, x * grad_output
setattr_value(Mul)

class Pow(Func):
    """
    >>> Value(2).pow(0)
    >>> Value(1)
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.saved_values.extend([x, y])
        return x ** y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_values
        grad1 = y * (x**(y - 1.0)) * grad_output
        grad2 = (x**y) * math.log(x) * grad_output if x > 0 else grad_output
        return grad1, grad2
setattr_value(Pow)

class Exp(Func):
    """
    >>> Value(1).exp()
    >>> Value(2.71828182846)
    """
    @staticmethod
    def forward(ctx, x):
        out = math.exp(x)
        ctx.saved_values.extend([out])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_values
        return grad_output * out
setattr_value(Exp)