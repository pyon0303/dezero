import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)
                
                      
class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]    
        ys = self.forward(xs)
        outputs = [Variable(self.as_array(y)) for y in ys]
        
        for outout in outputs:
            outout.set_creator(self)           
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
    def __str___(self):
        return type(self).__name__
    
    def as_array(self, x):
        if np.isscalar(x):
            return np.array(x)
        return x
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    
    
#for gradient checking
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    
    return (y1.data - y0.data) / (2 * eps)


def square(x):
    #callable
    return Square()(x)

def exp(x):
    return Exp()(x)


xs = [Variable(np.array(2.0)), Variable(np.array(3.0))]
f = Add()
ys = f(xs)
print(ys[0].data)