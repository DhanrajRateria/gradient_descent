import random
import pickle
from micrograd.node import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def l2_regularization(self, lambda_):
        return sum((p**2 for p in self.parameters())) * lambda_

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {p: 0 for p in params}
        self.v = {p: 0 for p in params}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, training=True):
        if training:
            mask = [Value(random.random() > self.p) for _ in x]
            return [xi * mi for xi, mi in zip(x, mask)]
        return x


def train_batch(model, optimizer, inputs, targets, batch_size):
    for i in range(0, len(inputs), batch_size):
        x_batch = inputs[i:i + batch_size]
        y_batch = targets[i:i + batch_size]
        
        # Forward pass
        preds = model(x_batch)
        loss = mse_loss(preds, y_batch)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)