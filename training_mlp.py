import struct
import time
import numpy as np
from matplotlib import pyplot as plt

from NN.layers.dense import Dense
from NN.layers.relu import ReLU
from NN.layers.softmax_crossentropy import SoftmaxCrossEntropy
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


# class MLP1(Model):
#     def __init__(self):
#         super().__init__()
#         self.layers = [
#             Dense(28 * 28, 10),
#             SoftmaxCrossEntropy()
#         ]
#
#
# class MLP2v1(Model):
#     def __init__(self):
#         super().__init__()
#         self.layers = [
#             Dense(28 * 28, 20),
#             ReLU(),
#             Dense(20, 10),
#             SoftmaxCrossEntropy()
#         ]
#
#
# class MLP2v2(Model):
#     def __init__(self):
#         super().__init__()
#         self.layers = [
#             Dense(28 * 28, 64),
#             ReLU(),
#             Dense(64, 10),
#             SoftmaxCrossEntropy()
#         ]


class MLP3v1(Model):
    def __init__(self):
        super().__init__()
        self.layers = [
            Dense(28 * 28, 10 * 10), ReLU(),
            Dense(10 * 10, 64), ReLU(),
            Dense(64, 10), SoftmaxCrossEntropy()
        ]



if __name__ == '__main__':
    # Hyperparams
    train_rat = 0.8
    epochs = 30
    batch_size = 16
    lr = 0.1
    metric_step = 1
    save_step = 5

    model = MLP3v1()
    optim = OptimizerOld(model, lr)

    # Data load
    test_images = read_idx3_ubyte('data/MNIST/t10k-images.idx3-ubyte')  # Читаем тестовые изображения
    test_labels = read_idx1_ubyte('data/MNIST/t10k-labels.idx1-ubyte')  # Читаем тестовые метки
    X = test_images.astype('float32') / 255.0  # Нормализация пикселей [0, 255] → [0, 1]

    # Изменение формы для полносвязной сети (10000, 28, 28) → (10000, 784)
    X = X.reshape(X.shape[0], -1)
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
    start = time.time()
    for epoch in range(epochs):
        X_train, T_train = shuffle_dataset(X_train, T_train)

        # Logging
        if epoch % metric_step == 0:
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

            print(
                f"Epoch:            {epoch}\n"
                f"Accuracy:         {round(accuracy * 100, 2)}%\n"
                f"Training Loss:    {train_loss:.2e}\n"
                f"Validation Loss:  {val_loss:.2e}"
            )
            print("Confusion Matrix:")
            print(confusion_matrix(model.predict(X_val), T_val))
            print()

        # Save params
        if epoch % save_step == 0:
            Y_val = model.predict(X_val)
            Y_train = model.predict(X_train)
            cm = confusion_matrix(Y_val, T_val)
            accuracy = round(np.trace(cm) / np.sum(cm), 2)
            train_loss = cross_entropy(Y_train, T_train)
            val_loss = cross_entropy(Y_val, T_val)
            with open(f"EP{epoch}_METRIX.txt", "w") as f:
                heading = (
                    f"[MODEL: MLP3v1]\n"
                    f"train ratio:      {train_rat}\n"
                    f"batch size:       {batch_size}\n"
                    f"learning rate:    {lr}\n"
                    f"------------------\n"
                    f"epoch:            {epoch} / {epochs}\n"
                    f"time:             {round(time.time() - start, 2)} s.\n"
                    f"accuracy:         {round(accuracy * 100, 2)}%\n"
                    f"Training Loss:    {train_loss:.2e}\n"
                    f"Validation Loss:  {val_loss:.2e}\n"
                )
                heading += "\n" + "[CONFUSION MATRIX]\n" + str(cm)
                f.write(heading)

            counter = 0
            for layer in model.layers:
                if layer.params:
                    counter += 1
                    W = layer.params[0]
                    b = layer.params[1]
                    np.savetxt(f"EP{epoch}_W{counter}", W)
                    np.savetxt(f"EP{epoch}_b{counter}", b)

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
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))  # два графика в ряд
    axs = axs.flatten()

    eps = [ep * metric_step for ep in range(epochs // metric_step)]

    # === График ошибок ===
    axs[0].set_title("Error per epoch")
    axs[0].set_xlabel("Epoch Number")
    axs[0].set_ylabel("Error")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].scatter(eps, training_losses, s=5, color="#483D8B", label="train error")
    axs[0].scatter(eps, validation_losses, s=5, color="#CD5C5C", label="val error")
    axs[0].legend()

    # === График точности ===
    axs[1].set_title("Accuracy per epoch")
    axs[1].set_xlabel("Epoch Number")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_ylim(0, 1)
    axs[1].scatter(eps, accuracies, s=1, color="#CD5C5C")

    # Текст
    plt.tight_layout()
    fig.text(
        x=0.01, y=0.05,
        s=f"Hyperparameters\n"
          f"Learning Rate: {lr:.2e}\n"
          f"Epoch Count: {epochs}\n"
          f"Training Ratio: {train_rat}\n"
          f"Batch Size: {batch_size}"
    )

    plt.subplots_adjust(bottom=0.2)
    plt.show()
    # # Открытие графика на весь экран
    # manager = plt.get_current_fig_manager()
    # manager.window.state('zoomed')
    #
    # # Сохранение картинки
    # plt.savefig("res.png", dpi=300, bbox_inches="tight")
