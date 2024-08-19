import numpy as np
import unittest
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero import utils
from numpy.testing import assert_array_equal

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


class Test37(unittest.TestCase):
    def test_reshpae(self):
        x = Variable(np.array([[1,2,3], [4,5,6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_flag=True)
        assert_array_equal(x.grad.data, np.ones(6).reshape(2,3))
        
    def test_reshape_from_variable(self):
        x = Variable(np.random.randn(1,2,3))
        y = x.reshape(2, 3)
        y = x.reshape((2, 3))
        
    def test_transpose(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        y = x.transpose()
        y.backward()
        assert_array_equal(x.grad.data, np.ones(6).reshape(2,3))
    
    def test_broadcast(self):
        x = np.array([1,2,3])
        y = np.broadcast_to(x, (2, 3))
        print(y)
        
    def test_sum_to(self):
        x = Variable(np.array([[1,2,3], [4,5,6]]))
        y = F.sum_to(x, (2, ))
        y.backward()
        assert_array_equal(x.grad.data, np.array([[1,1,1],[1,1,1]]))
    
    def test_broadcast_to(self):
        x = Variable(np.array([1,2,3]))
        y = F.broadcast_to(x, (2, 3))
        y.backward()
        assert_array_equal(x.grad.data, np.array([2, 2, 2]))
                