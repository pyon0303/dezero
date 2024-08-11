import numpy as np
import unittest
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero import utils

class Test31(unittest.TestCase):
    def test_sin(self):
        x = Variable(np.array(np.pi/2))
        y = F.sin(x)
        self.assertEqual(y.data, 1.0)
        y.backward(create_graph=True)
        gx = x.grad
        gx.backward(create_graph=True)
        self.assertAlmostEqual(x.grad.data, -1.0)
        
        
    def test_2nd_differentation(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y
        
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.0)
        
        gx = x.grad
        x.clear_grad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.0)
        
    def test_nth_differential_sin(self):
        x = Variable(np.linspace(-7,7,200))
        y = F.sin(x)
        y.backward(create_graph=True)
        
        for i in range(3):
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)
            print(x.grad)
            
    def test_tanh(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = 'x'
        y.name = 'y'
        y.backward(create_graph=True)
        
        iters = 5
        for i in range(iters):
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)
        
        gx = x.grad
        gx.name = 'gx' + str(iters+1)
        utils.plot_dot_graph(gx, verbose=False, to_file='tanh.png')
        
        
                