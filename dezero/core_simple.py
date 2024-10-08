import numpy as np
import weakref
import contextlib

class Config:
    enable_backprop = True
    
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__neg__ = neg

class Variable:
    
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
        
        
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
       return self.data.T
   
    def __len__(self) -> int:
        """
        returns the number of rows
        """
        return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)#len('variable(') == 9
        return 'variable(' + p + ')'
   
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None  
      
    def backward(self, retain_flag=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys) 
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
                
                if not retain_flag:
                    for y in f.outputs:
                        y().grad = None #y == weakref

                      
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]    
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for outout in outputs:
                outout.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
    def __str___(self):
        return type(self).__name__
    

class Square(Function):
    def forward(self, x):
        y =  x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
   
    def backward(self, gy):
        return gy, gy 
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return (y,)

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return (y,)
    
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return (y, )
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c
        
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        return self.c * self.inputs[0].data ** (self.c - 1) * gy

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * -np.cos(x)
        return gx
    
#for gradient checking
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    
    return (y1.data - y0.data) / (2 * eps)

def as_array(obj):
    if np.isscalar(obj):
        return np.array(obj)
    return obj

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def square(x):
    #callable
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x, y):
    y = as_array(y)
    return Add()(x, y)

def mul(x, y):
    y = as_array(y)
    return Mul()(x, y)

def neg(x):
    return Neg()(x)

def sub(x, y):
    y = as_array(y)
    return Sub()(x, y)

def rsub(x, y):
    y = as_array(y)
    return Sub()(y, x)

def div(x, y):
    y  = as_array(y)
    return Div()(x, y)

def rdiv(x, y):
    y = as_array(y)
    return Div()(y, x)

def pow(x, c):
    return Pow(c)(x)

def sin(x):
    return Sin()(x)