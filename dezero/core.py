import numpy as np
import weakref
import contextlib
import dezero

class Config:
    enable_backprop = True
    train = True
    
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

def test_mode():
    return using_config('train', False)

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
    Variable.__getitem__ = dezero.functions.get_item

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
       return dezero.functions.transpose(self)
   
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)
   
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
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self):
        return dezero.functions.transpose(self)
    
      
    def backward(self, retain_flag=False, create_graph=False):
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
            
            with using_config('enable_backprop', create_graph):
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
    
class Parameter(Variable):
    pass

class Square(Function):
    def forward(self, x):
        y =  x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs
        gx = 2 * x * gy
        return gx    

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
   
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape: # fit the shape between data and grad if broadcasted
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Pow(Function):
    def __init__(self, c):
        self.c = c
        
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x, = self.inputs
        c = self.c
        return c * x ** (c - 1) * gy


    
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
