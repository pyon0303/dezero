from typing import Any
from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
import os
from dezero import utils

class Layer:
    def __init__(self):
        self._params = set()
        
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
        
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
            
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj
                            
    def cleargrads(self):
        for param in self.params():
            param.clear_grad()
            
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        
    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
            
        y = F.linear(x, self.W, self.b)
        return y
    
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
    
        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
            
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
            
    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = utils.pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1] #channel
            self._init_W()
            
        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y
        
    

    
