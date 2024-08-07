import numpy as np
import unittest
from dezero.core import *

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