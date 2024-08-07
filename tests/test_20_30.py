import numpy as np
import unittest
from dezero import Variable

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