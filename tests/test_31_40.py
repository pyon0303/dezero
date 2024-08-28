import numpy as np
import unittest
import dezero
import dezero.datasets
import dezero.functions as F
import dezero.layers as L
import time
from dezero import Variable, Parameter, optimizers, utils, DataLoader
from dezero.models import TwoLayerNet, MLP
from numpy.testing import assert_array_equal

class Test31(unittest.TestCase):
    def test_sin(self):
        x = Variable(np.array(np.pi/2))
        y = F.sin(x)
        self.assertEqual(y.data, 1.0)
        y.backward(create_graph=True)
        gx = x.grad
        gx.backward(create_graph=True)
        self.assertAlmostEqual(x.grad.data, -1.0)
        
        
    def test_2nd_differentation(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y
        
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        self.assertEqual(x.grad.data, 24.0)
        
        gx = x.grad
        x.clear_grad()
        gx.backward()
        self.assertEqual(x.grad.data, 44.0)
        
    def test_nth_differential_sin(self):
        x = Variable(np.linspace(-7,7,200))
        y = F.sin(x)
        y.backward(create_graph=True)
        
        for i in range(3):
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)
            print(x.grad)
            
    def test_tanh(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = 'x'
        y.name = 'y'
        y.backward(create_graph=True)
        
        iters = 5
        for i in range(iters):
            gx = x.grad
            x.clear_grad()
            gx.backward(create_graph=True)
        
        gx = x.grad
        gx.name = 'gx' + str(iters+1)
        utils.plot_dot_graph(gx, verbose=False, to_file='tanh.png')


class Test37(unittest.TestCase):
    def test_reshpae(self):
        x = Variable(np.array([[1,2,3], [4,5,6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_flag=True)
        assert_array_equal(x.grad.data, np.ones(6).reshape(2,3))
        
    def test_reshape_from_variable(self):
        x = Variable(np.random.randn(1,2,3))
        y = x.reshape(2, 3)
        y = x.reshape((2, 3))
        
    def test_transpose(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        y = x.transpose()
        y.backward()
        assert_array_equal(x.grad.data, np.ones(6).reshape(2,3))
    
    def test_broadcast(self):
        x = np.array([1,2,3])
        y = np.broadcast_to(x, (2, 3))
        print(y)
        
    def test_sum_to(self):
        x = Variable(np.array([[1,2,3], [4,5,6]]))
        y = F.sum_to(x, (2, ))
        y.backward()
        assert_array_equal(x.grad.data, np.array([[1,1,1],[1,1,1]]))
    
    def test_broadcast_to(self):
        x = Variable(np.array([1,2,3]))
        y = F.broadcast_to(x, (2, 3))
        y.backward()
        assert_array_equal(x.grad.data, np.array([2, 2, 2]))
        
    def test_sum(self):        
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x, axis=1)
        y.backward()
        print(x.grad)
        
        x2 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y2 = F.sum(x2, axis=0)
        y2.backward()
        print(x2.grad)
        
    def test_matmul(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = F.matmul(x, W)
        y.backward()
        
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(W.grad.shape, W.shape)
        
    def test_broadcast_add(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_sub(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([-10]))
        y = x0 - x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_mul(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 * x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
    def test_broadcast_div(self):
        x0 = Variable(np.array([2,4,6]))
        x1 = Variable(np.array([2]))
        y = x0 / x1
        y.backward()
        print(x0.grad)
        print(x1.grad)
        
class Test44(unittest.TestCase):
    def test_layer(self):
        layer = L.Layer()
        
        layer.p1 = Parameter(np.array(1))
        layer.p2 = Parameter(np.array(2))
        layer.p3 = Variable(np.array(3))
        layer.p4 = 'test'
        
        print(layer._params)
        print('-----------')
        
        for name in layer._params:
            print(name, layer.__dict__[name])
    
            
    def test_linear(self):
        
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        
        l1 = L.Linear(10)
        l2 = L.Linear(1)
        
        def predict(x):
            y = l1(x)
            y = F.sigmoid(y)
            y2 = l2(y)
            return y2
        
        lr = 0.2
        iters = 10000
        
        for i in range(iters):
            y_pred = predict(x)
            loss = F.mean_squared_error(y_pred, y)

            l1.cleargrads()
            l2.cleargrads()
            loss.backward()
            
            for l in [l1, l2]:
                for p in l.params():
                    p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
                
    def test_Layer_composition(self):
        x = np.random.randn(100, 1)
        model = L.Layer()
        model.l1 = L.Linear(5)
        model.l2 = L.Linear(3)
        
        def predict(model, x):
            y = model.l1(x)
            y = F.sigmoid(y)
            y = model.l2(y)
            return y
        
        for p in model.params():
            print(p)
            
        model.cleargrads()
        
    def test_TwoLayerNet_plot(self):
        x = Variable(np.random.randn(5, 10), name='x')
        model = TwoLayerNet(100, 10)
        model.plot(x)
        
    def test_TwoLayerNet(self):
        lr = 0.2
        max_iter = 10000
        hidden_size = 10
        
        model = TwoLayerNet(hidden_size, 1)
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        
        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y_pred, y)
            
            model.cleargrads()
            loss.backward()
            
            for p in model.params():
                p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
                
    def test_MLP(self):
        lr = 0.2
        max_iter = 100000
        hidden_size = 10
        
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        model = MLP((10,20,30,40, 1))
        
        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y_pred, y)
            
            model.cleargrads()
            loss.backward()
            
            for p in model.params():
                p.data -= lr * p.grad.data
            
            if i % 1000 == 0:
                print(loss)
                
    def test_SGD(self):
        np.random.seed(0) 
        x = np.random.rand(100, 1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
        
        lr = 0.2
        max_iter = 20000
        hidden_size = 10
        
        model = MLP((hidden_size, 1))
        optimizer = optimizers.SGD(lr)
        optimizer.setup(model)
        
        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y, y_pred)
            
            model.cleargrads()
            loss.backward()
            
            optimizer.update()
            
            if i % 1000 == 0:
                print(loss)
                
    def test_np_add_at(self):
        a = np.zeros((2, 3))
        print(a)
        b = np.ones((3,))
        print(b)
        slices = 1
        np.add.at(a, slices, b) #a[indices] += b
        print(a)
        print(a[1][0:2])
    
    def test_get_item(self):
        a1 = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
        b = F.get_item(a1, 1)
        b.backward()
        print(a1.grad)
        
        a2 = Variable(np.array(([1, 2, 3],[4, 5, 6])))
        indices = np.array([0, 0, 1])
        y = F.get_item(a2, indices)
        print(y)
        
        a3 = Variable(np.array(([1, 2, 3],[4, 5, 6])))
        b3 = a3[:, 2]
        b3.backward()
        print(a3.grad)
        
    def test_soft_max(self):
        x = Variable(np.array([[0.2, -0.4]]))
        model = MLP((10, 3))
        y = model(x)
        p = F.softmax(y)
        print(y)
        print(p)
        
    def test_softmax_cross_entropy(self):
        x = np.array([[0.2, -0.4],[0.3,0.5],[1.3,-3.2],[2.1,0.3]])
        t = np.array([2,0,1,0])
        model = MLP((10, 3))
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        print(loss)
        
    def test_accuracy(self):
        y = np.array([[0.2,0.8,0], [0.1,0.1,0.8],[0.5,0.1,0.4]])
        t = np.array([1, 1, 0])
        acc = F.accuracy(y, t)
        self.assertAlmostEqual(acc.data, 0.6666, delta=1e-4)
        
    def test_spiral_with_dataloader(self):
        max_epoch = 300
        batch_size = 30
        hidden_size = 10
        lr = 1.0
        
        train_set = dezero.datasets.Spiral(train=True)
        test_set = dezero.datasets.Spiral(train=False)
        train_loader = dezero.DataLoader(train_set, batch_size, shuffle=True)
        test_loader = dezero.DataLoader(test_set, batch_size)
        
        model = MLP((hidden_size, 3))
        optimizer = optimizers.SGD(lr).setup(model)
        
        for epoch in range(max_epoch):
            sum_loss, sum_acc= 0, 0
            
            for x, t in train_loader:
                y = model(x)
                loss = F.softmax_cross_entropy_simple(y, t)
                acc = F.accuracy(y, t)
                model.cleargrads()
                loss.backward()
                optimizer.update()
                
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

            #train acc 0.9633
            print('epoch: {}'.format(epoch+1))
            print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(train_set), sum_acc/len(train_set)))
            
            sum_loss, sum_acc = 0, 0
            with dezero.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = F.softmax_cross_entropy_simple(y, t)
                    acc = F.accuracy(y, t)
                    sum_loss += float(loss.data) * len(t)
                    sum_acc += float(acc.data) * len(t)
            #test acc 0.9433    
            print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc/len(test_set)))
            
    def test_loading_mnist(self):
        train_set = dezero.datasets.MNIST(train=True)
        test_set = dezero.datasets.MNIST(train=False)
        
        print(len(train_set))
        print(len(test_set))
    
    def test_mnist(self):
        max_epoch = 5
        batch_size = 100
        hidden_size = 1000
        
        train_set = dezero.datasets.MNIST(train=True)
        test_set = dezero.datasets.MNIST(train=False)
        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size, shuffle=False)
        
        model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
        optimizer = optimizers.SGD().setup(model)
        
        for epoch in range(max_epoch):
            sum_loss, sum_acc = 0, 0
            
            for x, t in train_loader:
                y = model(x)
                loss = F.softmax_cross_entropy_simple(y, t)
                acc = F.accuracy(y, t)
                model.cleargrads()
                loss.backward()
                optimizer.update()
                
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
            #0.85
            print('epoch: {}'.format(epoch+1))
            print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(train_set), sum_acc/len(train_set)))
            
            sum_loss, sum_acc = 0, 0
            with dezero.no_grad():
                for x, t in test_loader:
                    y = model(x)
                    loss = F.softmax_cross_entropy_simple(y, t)
                    acc = F.accuracy(y, t)
                    
                    sum_loss += float(loss.data) * len(t)
                    sum_acc += float(acc.data) * len(t)
            #0.86
            print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc/len(test_set)))

    def test_save_load(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        data = {'x1': x1, 'x2': x2}
        np.savez('test.npz', **data)
        
        arrays = np.load('test.npz')
        x1 = arrays['x1']
        x2 = arrays['x2']
        print(x1)
        print(x2)

        #error happnes bacause weights is not created        
        model = MLP((1000, 10))
        filepath = 'test.npz'
        model.save_weights(filepath)
        
    def test_dropout(self):
        dropout_ratio = 0.6
        x = np.ones(10)
        xx = np.random.rand(10)
        print(xx)
        mask = xx > dropout_ratio
        y = x * mask
        print(y)
        
        scale = 1 - dropout_ratio
        y = x * scale
        print(y)
        
    def test_inverted_dropout(self):
        x = np.ones((10, 2))
        dropout_ratio = 0.6
        scale = 1 - dropout_ratio
        mask = np.random.rand(*x.shape)
        mask = mask > dropout_ratio
        y = x * mask / scale
        print(y)
    
    def test_get_conv_size(self):
        H, W = 4, 4
        KH, KW = 3, 3
        SH, SW = 1, 1
        PH, PW = 1, 1
        OH = utils.get_conv_outsize(H, KH, SH, PH)
        OW = utils.get_conv_outsize(W, KW, SW, PW)
        print(OH, OW)
        
    def test_img2col_max(self):
        x1 = np.arange(1, 28).reshape(1,3,3,3)
        co1 = F.im2col(x1, kernel_size=2, stride=1, pad=0, to_matrix=True)
        print(co1.shape)
        
        x2 = np.arange(1, 271).reshape(10, 3, 3, 3)
        kernel_size = (2, 2)
        stride = (1, 1)
        pad = (0, 0)
        col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
        print(col2.shape)
    
    def test_im2col_v1(self):
        x = np.arange(1, 17).reshape(4, 4)
        print(x, x.shape)
        
        def im2col(x, FH, FW):
            H, W = x.shape
            OH = H - FH + 1
            OW = W - FW + 1
            col = np.zeros((FH*FW, OH*OW))
            for h in range(OH):
                for w in range(OW):
                    test =  x[h:h+FH, w:w+FW]
                    col[:, w+h*OW] = test.reshape(-1)
                    
            return col
        
        print(im2col(x, 2, 2))
        
    def test_im2col_v2(self):
        def im2col(x, FH, FW):
            H, W = x.shape
            OH = H - FH + 1
            OW = W - FW + 1
            col = np.zeros((FH, FW, OH, OW))
            for h in range(FH):
                for w in range(FW):
                    col[h,w,:,:] = x[h:h+OH, w:w+OW]
            return col.reshape(FH*FW, OH*OW)
        
        x = np.arange(1, 17).reshape(4, 4)
        print(im2col(x, 2, 2))
    
    
    
    