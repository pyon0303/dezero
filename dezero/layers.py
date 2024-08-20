from typing import Any
from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F

class Layer:
    def __init__(self):
        self._params = set()
        
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)
        
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = tuple(outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
            
    def cleargrads(self):
        for param in self.params():
            param.clear_grad()
            
class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()
        
        I, O = in_size, out_size
        W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
        self.W = Parameter(W_data, name='W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(0, dtype=dtype), name='b')
        
    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y