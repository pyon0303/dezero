import unittest

class Test(unittest.TestCase):
    def test_add(self):
        expected = 1 + 2
        self.assertEqual(expected, 3)