import unittest
import sys, os
#sys.path.append(os.path.pardir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dezero.core import *  
import numpy as np

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
        