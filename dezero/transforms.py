import numpy as np

"TODO pip install PIL"

class Compose:
    """Compose several transforms.
    
    Args:
        transforms (lists): list of transforms
    
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms
        
    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
            return img
        
class Normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    
    def __call__(self, array):
        mean, std = self.mean, self.std
        
        return (array - mean) / std
    
class Astype:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
            
    def __call__(self, array):
        return array.astype(self.dtype)
            
                