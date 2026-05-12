import os
import struct
import time

from utils.backend import np
from matplotlib import pyplot as plt

from NN.layers.conv import Conv
from NN.layers.dense import Dense
from NN.layers.flatten import Flatten
from NN.layers.relu import ReLU
from NN.layers.softmax_crossentropy import SoftmaxCrossEntropy
from NN.layers.zeropad import ZeroPad
from NN.model import Model
from NN.optimizer import OptimizerOld
from NN.functions import shuffle_dataset, cross_entropy, confusion_matrix


def read_idx3_ubyte(filename):
    """
    Читает файл в формате idx3-ubyte (изображения MNIST)
    """
    with open(filename, 'rb') as f:
        # Читаем magic number
        magic_number = struct.unpack('>I', f.read(4))[0]

        # Читаем количество изображений
        num_images = struct.unpack('>I', f.read(4))[0]

        # Читаем размеры изображений
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        print(f"Magic number: {magic_number}")
        print(f"Number of images: {num_images}")
        print(f"Image size: {num_rows}x{num_cols}")

        # Читаем все пиксели
        buffer = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buffer, dtype=np.uint8)

        # Преобразуем в матрицу (num_images, rows, cols)
        images = data.reshape(num_images, num_rows, num_cols)

        return images


def read_idx1_ubyte(filename):
    """
    Читает файл в формате idx1-ubyte (метки MNIST)
    """
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]

        print(f"Magic number: {magic_number}")
        print(f"Number of items: {num_items}")

        buffer = f.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)

        return labels


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


def _zero_pad_corner(X: np.ndarray, p: int):
    if p == 0:
        return X
    N, K, H, W = X.shape
    zp = np.zeros((N, K, H + p, W + p))
    zp[:, :, :H, :W] = X
    return zp


class MLP1(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 10),
            SoftmaxCrossEntropy()
        ]


class MLP2v1(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 20),
            ReLU(),
            Dense(20, 10),
            SoftmaxCrossEntropy()
        ]


class MLP2v2(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 64),
            ReLU(),
            Dense(64, 10),
            SoftmaxCrossEntropy()
        ]


class MLP3v1(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 10 * 10), ReLU(),
            Dense(10 * 10, 64), ReLU(),
            Dense(64, 10), SoftmaxCrossEntropy()
        ]


class MLP3v2(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 24 * 24), ReLU(),
            Dense(24 * 24, 20 * 20), ReLU(),
            Dense(20 * 20, 10 * 10), ReLU(),
            Dense(10 * 10, 10),
            SoftmaxCrossEntropy()
        ]


# 97%
class CNNv1(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            Conv(r=8, r_prev=1, n=3),  # [80 params]
            ReLU(),
            Flatten(),
            # -> (N, 8 * 26 * 26)
            Dense(8 * 26 * 26, 128),  # [692352 params]
            ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv2(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28) [10 params]
            Conv(r=1, r_prev=1, n=3),
            ReLU(),
            # -> (N, 1, 26, 26)
            Conv(r=8, r_prev=1, n=3),  # [80 params]
            ReLU(),
            # -> (N, 8, 24, 24)
            Flatten(),
            # -> (N, 8 * 24 * 24)
            Dense(8 * 24 * 24, 128),  # [2_359_808 params]
            ReLU(),
            # -> (N, 4096)
            Dense(128, 10),  # [5_130 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv3(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            ZeroPad(right=1, bottom=1, left=0, top=0),
            # -> (N, 1, 29, 29)
            Conv(r=1, r_prev=1, n=3, s=1),  # [10 params]
            ReLU(),
            # -> (N, 1, 27, 27)
            Conv(r=8, r_prev=1, n=3, s=1),  # [80 params]
            ReLU(),
            # -> (N, 8, 25, 25)
            Conv(r=16, r_prev=8, n=3, s=2),  # [1_168 params]
            ReLU(),
            # -> (N, 16, 12, 12)
            Flatten(),
            # -> (N, 16 * 12 * 12)
            Dense(16 * 12 * 12, 128),  # [295_040 params]
            ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv4(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 29, 29)
            Conv(r=16, r_prev=1, n=3, s=1),  # [160 params]
            ReLU(),
            # -> (N, 16, 27, 27)
            Conv(r=8, r_prev=16, n=3, s=1),  # [80 params]
            ReLU(),
            # -> (N, 8, 25, 25)
            Conv(r=4, r_prev=8, n=3, s=2),  # [1_168 params]
            ReLU(),
            # -> (N, 4, 12, 12)
            Flatten(),
            # -> (N, 4 * 12 * 12)
            Dense(4 * 12 * 12, 128),  # [295_040 params]
            ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv5(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 29, 29)
            Conv(r=2, r_prev=1, n=3),  # [20 params]
            ReLU(),
            # -> (N, 2, 27, 27)
            Conv(r=2, r_prev=2, n=3),  # [38 params]
            ReLU(),
            # -> (N, 2, 25, 25)
            Conv(r=4, r_prev=2, n=3, s=2),  # [76 params]
            ReLU(),
            # -> (N, 4, 12, 12)
            Conv(r=8, r_prev=4, n=3),  # [296 params]
            ReLU(),
            # -> (N, 8, 10, 10)
            Conv(r=8, r_prev=8, n=3),  # [584 params]
            ReLU(),
            # -> (N, 8, 8, 8)
            Conv(r=16, r_prev=8, n=2, s=2),  # [528 params]
            ReLU(),
            # -> (N, 16, 4, 4)
            Flatten(),
            # -> (N, 16 * 4 * 4)
            Dense(16 * 4 * 4, 128),  # [32896 params]
            ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv6(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            Conv(r=2, r_prev=1),
            # -> (N, 2, 26, 26)
            ZeroPad(),
            # -> (N, 2, 28, 28)
            ReLU(),
            # -> (N, 2, 28, 28)
            Conv(r=4, r_prev=2), ReLU(),
            # -> (N, 4, 26, 26)
            Flatten(),
            # -> (N, 4 * 26 * 26)
            Dense(4 * 26 * 26, 128), ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv7(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            ZeroPad(),
            # -> (N, 1, 30, 30)
            Conv(r=2, r_prev=1),
            # -> (N, 2, 28, 28)
            ZeroPad(),
            # -> (N, 2, 30, 30)
            ReLU(),
            # -> (N, 2, 30, 30)
            Conv(r=4, r_prev=2, s=3),
            # -> (N, 4, 10, 10)
            ReLU(),
            # -> (N, 4, 10, 10)
            Flatten(),
            # -> (N, 4 * 10 * 10)
            Dense(4 * 10 * 10, 128), ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv8(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            ZeroPad(),
            # -> (N, 1, 30, 30)
            Conv(r=2, r_prev=1), ReLU(), ZeroPad(),
            # -> (N, 2, 30, 30)
            Conv(r=4, r_prev=2), ReLU(),
            # -> (N, 4, 28, 28)
            Conv(r=8, r_prev=4, n=2, s=2), ReLU(), Flatten(),
            # -> (N, 8, 14, 14)
            Dense(8 * 14 * 14, 128), ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


class CNNv9(Model):  # 97%
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            ZeroPad(),
            # -> (N, 1, 30, 30)
            Conv(r=2, r_prev=1), ReLU(), ZeroPad(),
            # -> (N, 2, 30, 30)
            Conv(r=4, r_prev=2), ReLU(), ZeroPad(),
            # -> (N, 4, 30, 30)
            Conv(r=4, r_prev=4), ReLU(), Flatten(),
            # -> (N, 4, 28, 28)
            Dense(4 * 28 * 28, 128), ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]

class CNNv10(Model):  # 97%
    def __init__(self):
        super().__init__()
        self.layers = [
            # -> (N, 1, 28, 28)
            ZeroPad(),
            # -> (N, 1, 30, 30)
            Conv(r=4, r_prev=1), ReLU(), ZeroPad(),
            # -> (N, 4, 30, 30)
            Conv(r=8, r_prev=4), ReLU(), ZeroPad(),
            # -> (N, 8, 30, 30)
            Conv(r=8, r_prev=8), ReLU(), ZeroPad(right=1, bottom=1, top=0, left=0),
            # -> (N, 8, 29, 29)
            Conv(r=8, r_prev=8, n=3, s=2), ReLU(),

            # -> (N, 8, 14, 14)
            Conv(r=16, r_prev=8), ReLU(), ZeroPad(),
            # -> (N, 16, 14, 14)
            Conv(r=32, r_prev=16), ReLU(), ZeroPad(),
            # -> (N, 32, 14, 14)
            Conv(r=32, r_prev=32), ReLU(), ZeroPad(right=1, bottom=1, top=0, left=0),
            # -> (N, 32, 13, 13)
            Conv(r=32, r_prev=32, n=3, s=2), ReLU(),
            # -> (N, 32, 6, 6)


            Flatten(),
            Dense(32 * 6 * 6, 128), ReLU(),
            # -> (N, 128)
            Dense(128, 10),  # [1_290 params]
            SoftmaxCrossEntropy()
            # -> (N, 10)
        ]


if __name__ == '__main__':
    model = CNNv10()

    # Hyperparams
    train_rat = 0.8
    epochs = 15
    batch_size = 16
    lr = 0.01
    # metric_step = 1
    save_step = 1

    optim = OptimizerOld(model, lr)

    # Data load
    test_images = read_idx3_ubyte('data/MNIST/t10k-images.idx3-ubyte')  # Читаем тестовые изображения
    test_labels = read_idx1_ubyte('data/MNIST/t10k-labels.idx1-ubyte')  # Читаем тестовые метки
    X = test_images.astype('float32') / 255.0 - 1 / 2  # Нормализация пикселей [0, 255] → [-0.5, 0.5]

    # Изменение формы для полносвязной сети (10000, 28, 28) → (10000, 784)
    X = X.reshape(-1, 1, 28, 28)
    T = one_hot_encode(test_labels)
    X, T = shuffle_dataset(X, T)

    # Data split
    X_train = X[:int(train_rat * len(X))]
    T_train = T[:int(train_rat * len(T))]
    X_val = X[int(train_rat * len(X)):]
    T_val = T[int(train_rat * len(T)):]

    # Metrics
    training_losses = []
    validation_losses = []
    accuracies = []
    confusion_matrices = []

    # Train
    if not os.path.exists(model.__class__.__name__):
        os.makedirs(model.__class__.__name__)
    start = time.time()
    for epoch in range(epochs):
        # Перемешиваем датасет
        X_train, T_train = shuffle_dataset(X_train, T_train)

        # Logging & Save
        if epoch % save_step == 0:
            Y_val = model.predict(X_val)
            Y_train = model.predict(X_train)
            cm = confusion_matrix(Y_val, T_val)
            accuracy = float(np.trace(cm) / np.sum(cm))
            train_loss = float(cross_entropy(Y_train, T_train))
            val_loss = float(cross_entropy(Y_val, T_val))
            confusion_matrices.append(cm)
            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            accuracies.append(accuracy)
            log = (
                f"Hyperparams:      train ratio = {train_rat}, batch size = {batch_size}, learning rate = {lr}\n"
                f"Epoch:            {epoch} / {epochs}\n"
                f"Time:             {round(time.time() - start, 2)} s.\n"
                f"Accuracy:         {round(accuracy * 100, 2)}%\n"
                f"Overfit:          {round(100 - train_loss / val_loss * 100, 2)}%\n"
                f"Training Loss:    {train_loss:.2e}\n"
                f"Validation Loss:  {val_loss:.2e}"
            )

            print(log)
            with open(f"{model.__class__.__name__}/EP_{epoch}_METRICS.txt", "w") as f:
                f.write(log + "\n" + "[CONFUSION MATRIX]\n" + str(cm))

            model.save_params(f"{model.__class__.__name__}/EP_{epoch}_PARAMS.npz")
            print("PARAMS SAVED")

        # Batch iter
        for i in range(X_train.shape[0] // batch_size):
            X_batch = X_train[batch_size * i: batch_size * (i + 1)]
            T_batch = T_train[batch_size * i: batch_size * (i + 1)]

            # Forward
            Y_batch = model.forward(X_batch)

            # Backward
            optim.backward(T_batch)

            # Update params
            optim.step()

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))  # два графика в ряд
    axs = axs.flatten()

    eps = [ep * save_step for ep in range(epochs // save_step)]

    # === График ошибок ===
    axs[0].set_title("Error per epoch")
    axs[0].set_xlabel("Epoch Number")
    axs[0].set_ylabel("Error")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].scatter(eps, training_losses, s=5, color="#483D8B", label="train error")
    axs[0].scatter(eps, validation_losses, s=5, color="#CD5C5C", label="val error")
    axs[0].legend()

    # === График точности ===
    axs[1].set_title("Metrics")
    axs[1].set_xlabel("Epoch Number")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_ylim(0, 1)
    axs[1].scatter(eps, accuracies, s=5, color="#CD5C5C", label="accuracy")
    axs[1].scatter(eps, [1 - tr / val for tr, val in zip(training_losses, validation_losses)],
                   s=5, color="blue", label="train err / val err")
    axs[1].legend()

    # Текст
    plt.tight_layout()
    fig.text(
        x=0.01, y=0.2,
        s=f"Hyperparameters\n"
          f"Learning Rate: {lr:.2e}\n"
          f"Epoch Count: {epochs}\n"
          f"Training Ratio: {train_rat}\n"
          f"Batch Size: {batch_size}"
    )
    fig.text(
        x=0.66, y=0.2,
        s=f"Params\n"
          f"{[l.params[0].size + l.params[1].size if l.params else 0 for l in model.layers]}"
    )
    t_tot = time.time() - start

    fig.text(
        x=0.33, y=0.2,
        s=f"Time\n"
          f"Total: {round(t_tot, 2)} s.\n"
          f"Per epoch: {round(t_tot / epochs, 2)}"
    )

    plt.subplots_adjust(bottom=0.5)
    # Открытие графика на весь экран
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')

    # Сохранение картинки
    plt.savefig(f"{model.__class__.__name__}/TRAIN.png", dpi=300, bbox_inches="tight")
    plt.show()
