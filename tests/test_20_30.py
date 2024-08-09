import numpy as np
import unittest
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import utils


class Test20(unittest.TestCase):
    def test_overload_1(self):
        a = Variable(np.array(3))
        b = Variable(np.array(2))
        c = Variable(np.array(1.0))
        y = a * b + c
        y.backward()        
        self.assertEqual(a.grad, 2)
        self.assertEqual(b.grad, 3)
    
    def test_overload_2(self):
        a = Variable(np.array(3))
        b = a + 2
        c = 2 + a
        d = a * 3
        e = 3 * a
        self.assertEqual(b.data, 5.0)
        self.assertEqual(c.data, 5.0)
        self.assertEqual(d.data, 9.0)
        self.assertEqual(e.data, 9.0)
        f = np.array(2.0) + a
        self.assertEqual(np.array(1).__array_priority__, 0.0)
        self.assertEqual(f.data, 5.0)
        
    def test_overload_neg_sub(self):
        a = Variable(np.array(3))
        b = -a
        self.assertEqual(b.data, -3.0)
        c = a - 2
        d = 2 - a
        self.assertEqual(c.data, 1.0)
        self.assertEqual(d.data, -1.0)
         
    def test_overload_div_rdiv_pow(self):
        a = Variable(np.array(5))
        b = a / 3
        c = 3 / a
        d = a ** 3
        self.assertAlmostEqual(b.data, 5/3)
        self.assertAlmostEqual(c.data, 3/5)
        self.assertEqual(d.data, 125)
      
        
class Test24(unittest.TestCase):
    def test_sphere(self):
        def sphere(x, y) -> Variable:
            return x ** 2 + y ** 2
        
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(x.grad, 2.0)
        self.assertEqual(y.grad, 2.0)
        
    def test_matyas(self):
        def matyas(x, y) -> Variable:
            return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        print(x.grad, y.grad)
        self.assertAlmostEqual(x.grad, 0.04)
        self.assertAlmostEqual(y.grad, 0.04)
        
    def test_goldstein(self):
        def goldstein(x, y):
            z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
                   (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
            return z
            
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        print(x.grad, y.grad)
        self.assertAlmostEqual(x.grad, -5376.0)
        self.assertAlmostEqual(y.grad, 8064.0)
        

class Test26(unittest.TestCase):
    def test_dot_var(self):
        x = Variable(np.random.randn(2, 3))
        x.name = 'x'
        print(utils._dot_var(x))
        print(utils._dot_var(x, verbose=True))
        
    def test_dot_func(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y = x0 + x1
        txt = utils._dot_func(y.creator)
        print(txt)
        
    def test_plot_dot_graph(self):
        def goldstein(x, y):
            z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
                   (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
            return z
            
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        
        x.name = 'x'
        y.name = 'y'
        z.name = 'z'
        utils.plot_dot_graph(z, verbose=False, to_file='goldstein.png')


