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