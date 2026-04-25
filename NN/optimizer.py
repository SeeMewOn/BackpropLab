from utils.backend import np

from NN.model import Model


class Optimizer:
    def __init__(
            self,
            model: Model,
            lr: float,  # learning rate
    ):
        self.layers_reversed = list(reversed(model.layers))
        self.lr = lr

    def backward(self, T: np.ndarray):
        """ Вычисление градиентов по обучаемым параметрам и сохранение оных """
        out = T  # (56)
        for layer in self.layers_reversed:
            out = layer.backward(out)

    def step(self):
        """ Params update """
        for layer in self.layers_reversed:
            #
            if layer.params:
                layer.params[0] -= self.lr * layer.grads[0]
                layer.params[1] -= self.lr * layer.grads[1]
