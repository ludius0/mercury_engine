# libs
from numbers import Complex

class Value:
    """
    Numercial class perfoming mathematical operations and saving parental history
    for computing gradient of derivation.

    Example:
    >>> a = Value(-2)
    >>> b = Value(3)
    >>> c = a * 2 + (a**3).relu() - Value(2).add(b.exp()) + 1
    >>> c
    >>> 23
    >>> c.backward()
    >>> b.grad
    >>> -20.085536923187668
    """
    def __init__(self, data, grad=None, _ctx=None):
        assert isinstance(data, Complex) 
        assert isinstance(grad, Complex) or grad == None
        self.data = data
        self.grad = grad
        self._ctx = _ctx

    def __repr__(self):
        return f"Value({self.data})"

    def __format__(self,fmt):       
        return f'Value({self.data:{fmt}})'

    def __neg__(self):
        return self * -1
    
    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        return self.pow(other)
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def _save_for_derivation(self, *args):
        self.save_for_derivative.extend(*args)
    
    def backward(self, allow_fill=True):
        if self._ctx is None: return # no history
        if self.grad is None and allow_fill: self.grad = 1 # create gradient 1 for first one (one using this func)
        assert (self.grad is not None)
        """"
        Build ordered list of history of all computations of Value classes.
        """
        topo = []   # topological order
        visited = set()
        def build_topo(x):
            visited.add(x)
            if x._ctx is not None:
                for child in x._ctx.parents:
                    if child not in visited:
                        build_topo(child)
                topo.append(x)
        build_topo(self)

        for value in reversed(topo):
            """
            Looping back and for each Value() involved in mathematical computing
            to compute its gradient.
            """
            grads = value._ctx.backward(value._ctx, value.grad)
            if not isinstance(grads, tuple): grads = (grads,) # for iteration
            for v, g in zip(value._ctx.parents, grads):
                if g is None:   continue # skip ones without parents
                v.grad = g if v.grad is None else (v.grad + g)