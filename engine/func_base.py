# scripts
from .value_base import Value

class Func:
    """
    Backbone of every class used for mathematical computing (in op_math.py).
    During forward() it will store data for later use case of backward().
    """
    def __init__(self, *values, **kwargs):
        self.parents = values
        self.saved_values = []
    
    def later2backward(self, *values):
        self.saved_values.extend(values)

def setattr_value(cls):
    """ 
    'setattr_tensor(cls)' assign child class (like in op_math.py)
    of Func class as in-build function of Value class (value_base.py).

    So for example the class Add (from op_math.py) is inherited in the Value class
    as 'Value().add()' and it use call_func(*args, **kwargs),
    which automatically do:
        1. check if all arguments (*args) are Value class
        2. apply forward() of operational function 'op func' (child class of Func class), 
            this return a new Value class. (Additionaly first input is 'it self' for storing
            data for backward() (computing derivative))
        3. insert op func into the new Value class (in _ctx) 
            for future reference when computing gradient 
            (using backward() of op func for Value().backward())
        4. return the new Value()
    """
    # cls -> operation (op)
    def call_func(*args, **kwargs):
        # check if each arg is Value() (if not then convert to one)
        # so operations like "Value(10) + 5" can work
        parents = tuple(t if isinstance(t, Value) else Value(t) for t in args)
        # computing forward of cls (op func)
        ctx = cls(*parents) # for later2backward()
        ret = Value(cls.forward(ctx, *[t.data for t in parents], **kwargs))
        ret._ctx = ctx # save for backward()
        return ret
    setattr(Value, cls.__name__.lower(), call_func)