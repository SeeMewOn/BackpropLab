from utils.backend import np
from matplotlib import pyplot as plt

from training import CNNv3, read_idx3_ubyte, read_idx1_ubyte, one_hot_encode, CNNv8

if __name__ == '__main__':
    model = CNNv8()
    model.load_params("CNNv8/EP_13_PARAMS.npz")

    # Data load
    test_images = read_idx3_ubyte('data/MNIST/t10k-images.idx3-ubyte')  # Читаем тестовые изображения
    test_labels = read_idx1_ubyte('data/MNIST/t10k-labels.idx1-ubyte')  # Читаем тестовые метки
    X = test_images.astype('float32') / 255.0 - 1 / 2  # Нормализация пикселей [0, 255] → [-0.5, 0.5]
    X = X.reshape(-1, 1, 28, 28)
    T = one_hot_encode(test_labels)

    # Предсказание
    N = 666
    print(model.predict(np.array([X[N]])).argmax())

    # plt.imshow(X[N][0])
    # plt.show()

    x_target: np.ndarray = np.zeros((2, 30, 30)).astype(np.float32)
    t_target: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    X_target = np.array([x_target])
    T_target = np.array([t_target])
    # print("DDD")
    # print(X_target.shape)
    # print(T_target.shape)
    lr = 0.01
    epochs = 100
    layers = model.layers[4:]
    for ep in range(epochs):
        # Forward
        out = X_target
        for layer in layers:
            out = layer.forward(out)
        print(f"iter: {ep}")
        # Y_target = model.forward(X_target)
        print(round(float(np.linalg.norm(out - T_target)), 3))
        # print(Y_target.argmax())
        out = T_target
        for layer in reversed(layers):
            out = layer.backward(out, calc_grads=False)

        X_target -= lr * out
        # # --- Трюки для красоты ---
        # X_target *= 0.99  # L2 регуляризация (обязательно!)
        #
        # # Ограничение диапазона
        # X_target = np.clip(X_target, -0.5, 0.5)
        # # Раз в 10 итераций можно чуть-чуть "встряхнуть" картинку шумом,
        # # чтобы она не застревала в локальных минимумах-кратерах
        # if ep % 10 == 0:
        #     X_target += np.random.randn(*X_target.shape) * 0.005

    # print(model.predict(X_target).argmax())
    # .get() переносит массив с GPU на CPU
    im1 = X_target[0][1].get() if hasattr(X_target, 'get') else X_target[0][0]
    im2 = X_target[0][0].get() if hasattr(X_target, 'get') else X_target[0][0]

    plt.imshow(im1, cmap='gray')
    plt.show()
    plt.imshow(im2, cmap='gray')

    plt.show()
