import numpy as np
import gzip
from dezero.transforms import Compose
from dezero.transforms import Flatten, Normalize, ToFloat
from dezero.utils import get_file

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()
            
    def prepare(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None: # for non-supervisial
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

# =========================================================================================
# Toy datasets
# =========================================================================================

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class BigData(Dataset):
    def __getitem__(self, index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label{}.npy'.format(index))
        return x, t
    
    def __len__():
        return 100000
    
class MNIST(Dataset):
    def __init__(self, train=True,transform=Compose([Flatten(), ToFloat(),
                                                     Normalize(0.,255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)
    
    def prepare(self):
        #url https://yann.lecun.com/exdb/mnist/
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}
        
        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])
        
        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)
        
    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28) #size, channel, width, height
        return data

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels
    

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed)
    
    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=int)
    
    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j
    
    #shuffle    
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels

