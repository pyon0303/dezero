import unittest
from dezero import Variable
from dezero.core_simple import square, add, Config
import numpy as np
from numpy.testing import assert_array_equal

class Test13(unittest.TestCase):
    def test_addgrad(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(x.grad, 4.0)
        self.assertEqual(y.grad, 6.0)
        self.assertEqual(z.data, 13.0)
        

class Test14(unittest.TestCase):
    def test_add_same_variable(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        self.assertEqual(y.data, 6.0)
        
        y.backward()
        self.assertEqual(x.grad, 2.0)
        
        x.clear_grad()
        y = add(add(x,x), x)
        y.backward()
        self.assertEqual(x.grad, 3.0)
        

class Test15(unittest.TestCase):
    def test_check_switch_config(self):
        with no_grad():
            x = Variable(np.array(5.0))
            y = square(x)
            self.assertEqual(Config.enable_backprop, False)
            
        self.assertEqual(Config.enable_backprop, True)
    
    def test_topology(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        
        y.backward()
        self.assertEqual(x.grad, 64.0)
    
    def test_my_sqrt(self):
        prev = 9
        while True:
            next = ((prev * prev) + 9) / (2*prev)
            if prev - next < 1e-09:
                return next
            prev = next
            
    
    def test_weak_ref(self):
        x0 = Variable(np.array(5.0))
        x1 = Variable(np.array(5.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        
        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x0.grad)
        
    
    def test_enable_backprop_false(self):
        Config.enable_backprop = False
        x0 = Variable(np.array(5.0))
        x1 = Variable(np.array(10.0))
        x2 = add(x0, x1)
        with self.assertRaises(AttributeError):
            x2.backward()
            
        

class Test19(unittest.TestCase):
    def test_properties(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.dtype, np.int64)
        self.assertEqual(x.size, 6)
        assert_array_equal(x.T, np.array([[1,4],[2,5],[3,6]]))
        self.assertEqual(len(x), 2)
        print(x)

     
        