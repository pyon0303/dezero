import numpy as np
import unittest
import matplotlib.pyplot as plt
from dezero import Variable, Parameter
import dezero.functions as F
import dezero.layers as L
from dezero.models import TwoLayerNet, MLP
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
        
    def test_sum(self):        
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x, axis=1)
        y.backward()
        print(x.grad)
        
        x2 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y2 = F.sum(x2, axis=0)
        y2.backward()
        print(x2.grad)
        
    def test_matmul(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = F.matmul(x, W)
        y.backward()
        
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(W.grad.shape, W.shape)
        
    def test_broadcast_add(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_sub(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([-10]))
        y = x0 - x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_mul(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_div(self):
        x0 = Variable(np.array([2,4,6]))
        x1 = Variable(np.array([2]))
        y = x0 / x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
class Test44(unittest.TestCase):
    def test_layer(self):
        layer = L.Layer()
        
        layer.p1 = Parameter(np.array(1))
        layer.p2 = Parameter(np.array(2))
        layer.p3 = Variable(np.array(3))
        layer.p4 = 'test'
        
        print(layer._params)
        print('-----------')
        
        for name in layer._params:
            print(name, layer.__dict__[name])
    
            
    def test_linear(self):
        
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        
        l1 = L.Linear(10)
        l2 = L.Linear(1)
        
        def predict(x):
            y = l1(x)
            y = F.sigmoid(y)
            y2 = l2(y)
            return y2
        
        lr = 0.2
        iters = 10000
        
        for i in range(iters):
            y_pred = predict(x)
            loss = F.mean_squared_error(y_pred, y)

            l1.cleargrads()
            l2.cleargrads()
            loss.backward()
            
            for l in [l1, l2]:
                for p in l.params():
                    p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
                
    def test_Layer_composition(self):
        x = np.random.randn(100, 1)
        model = L.Layer()
        model.l1 = L.Linear(5)
        model.l2 = L.Linear(3)
        
        def predict(model, x):
            y = model.l1(x)
            y = F.sigmoid(y)
            y = model.l2(y)
            return y
        
        for p in model.params():
            print(p)
            
        model.cleargrads()
        
    def test_TwoLayerNet_plot(self):
        x = Variable(np.random.randn(5, 10), name='x')
        model = TwoLayerNet(100, 10)
        model.plot(x)
        
    def test_TwoLayerNet(self):
        lr = 0.2
        max_iter = 10000
        hidden_size = 10
        
        model = TwoLayerNet(hidden_size, 1)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        
        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y_pred, y)
            
            model.cleargrads()
            loss.backward()
            
            for p in model.params():
                p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
                
    def test_MLP(self):
        lr = 0.2
        max_iter = 100000
        hidden_size = 10
        
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        model = MLP((10,20,30,40, 1))
        
        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y_pred, y)
            
            model.cleargrads()
            loss.backward()
            
            for p in model.params():
                p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
        
        