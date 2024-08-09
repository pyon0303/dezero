import numpy as np
import unittest
import matplotlib.pyplot as plt
from dezero import Variable
from dezero.core_simple import sin

class Test31(unittest.TestCase):
    def test_sin(self):
        x = Variable(np.array(np.pi/2))
        y = sin(x)
        self.assertEqual(y.data, 1.0)
        
        y.backward(retain_flag=True)
        self.assertEqual(y.grad, 1.0)
        self.assertAlmostEqual(x.grad, 0.0)
        