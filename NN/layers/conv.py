from utils.backend import np
from utils.backend import as_strided
from NN.layer import Layer

# TODO добавить возможность не учитывать смещения. Это может пригодиться, например, когда после Conv стоит BatchNorm
class Conv(Layer):
    def __init__(
            self,
            r: int,
            r_prev: int,
            n: int = 3,
            # m: int = 3,
            s: int = 1,
            # s_m: int = 1,
    ):
        """
        Свёрточный слой.
        # :param m: Количество строк свёрточного фильтра.
        :param n: Количество столбцов свёрточного фильтра.
        :param r_prev: Глубина свёрточного фильтра.
        :param r: Количество свёрточных фильтров.
        :param s: Страйд свёртки.
        """
        super().__init__()
        r_in = r_prev * n * n
        W: np.ndarray = np.random.randn(r, r_prev, n, n, ).astype(np.float32) * np.sqrt(2.0 / r_in)
        b: np.ndarray = np.zeros(r).astype(np.float32)

        self.params = [W, b]
        self.s = s
        self.Z_prev: np.ndarray = np.array([])

    def predict(self, X: np.ndarray) -> np.ndarray:
        W, b = self.params
        B = b.reshape(1, -1, 1, 1)
        A = _conv2d_batch_ext(X, W, self.s) + B  # Это активация по батчу!!! (N, K, H_out, W_out)

        return A

    def forward(self, X: np.ndarray) -> np.ndarray:
        # if self.p > 0:
        #     X = _zero_pad_corner(X, self.p)
        self.Z_prev = X
        return self.predict(X)

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        W = self.params[0]
        K, C, H_f, W_f = W.shape
        assert H_f == W_f  # Пока что тут постоит костыль. TODO может быть в дальнейшем обобщить

        # Получаем dL_dZ_prev
        zpd_dL_dA = _zero_pad(_dilate(dL_dA, self.s - 1), W_f - 1)

        U = _w2u(W)
        dL_dZ_prev = _conv2d_batch_ext(zpd_dL_dA, U)

        if self.is_training:
            # Получаем dL_dW
            Z_trans = self.Z_prev.transpose(1, 0, 2, 3)
            C, N, HZ, WZ = Z_trans.shape
            dil_dL_dA = _dilate(dL_dA.transpose(1, 0, 2, 3), self.s - 1)
            dL_dW = _conv2d_batch_ext(Z_trans, dil_dL_dA).transpose(1, 0, 2, 3) / N

            # Получаем dL_db
            dL_db = np.sum(dL_dA, axis=(0, 2, 3)) / N

            # Сохраняем градиенты и возвращаем результат
            self.grads = [dL_dW, dL_db]
        return dL_dZ_prev


def _conv2d_ext(
        x: np.ndarray,
        W: np.ndarray,
        stride=1,
) -> np.ndarray:
    """
    [Расширенная двумерная свёртка]
    :param x: (C, H_x, W_x) Вход (картинка).
    :param W: (K, C, H_f, W_f) - фильтры (K штук).
    :param stride: Страйд свёртки.
    :return: Тензор
    """
    C, H_x, W_x = x.shape
    K, C_f, H_f, W_f = W.shape
    assert C == C_f, "Количество каналов входа и фильтра должны совпадать"

    # Размеры выхода
    H_out = (H_x - H_f) // stride + 1
    W_out = (W_x - W_f) // stride + 1

    # Получаем текущие strides для входного тензора.
    # Для float32 это (height * width * 4, width * 4, 4)
    s_c, s_h, s_w = x.strides

    # Создаём виртуальную матрицу окон размером new_shape = (H_out, W_out, C, H_f, W_f)
    #
    # Воспринимаем этот 5D тензор как 2D матрицу, элементы которой - 3D кубики (окна).
    # Умножая по Адамару фильтр w (айтем из W) на окно и складывая элементы
    # результата получим соответствующий элемент свёртки.
    # Исходя из такой логики можем построить new_shape. Так как размерность new_shape равна 5,
    # то и размерность new_strides будет равна 5.
    #
    # Когда я прикладываю фильтр к картинке я умножаю w[c, i, j] на x[c, y+i, x+i],
    # где (x, y) - координаты верхнего левого угла фильтра (не путать координату x с картинкой x).
    # Внутри окна нужно перемещаться между соседними пикселями точно так же как в исходной картинке:
    #
    # - чтобы перейти от x[0, 0, 0] к x[0, 0, 1] нужно прыгнуть на s_w байт
    # - чтобы перейти от x[0, 0, 0] к x[0, 1, 0] нужно прыгнуть на s_h байт
    # - чтобы перейти от x[0, 0, 0] к x[1, 0, 0] нужно прыгнуть на s_c байт
    #
    # Поэтому последние 3 страйда в new_shape будут (s_c, s_h, s_w).
    #
    # - чтобы начать читать в памяти следующее окно по горизонтали нужно сдвинуться на stride * s_w
    # - чтобы начать читать в памяти следующее окно по вертикали нужно сдвинуться на stride * s_h
    #
    # stride тут - это страйд свёртки, который по умолчанию в функции равен 1!
    new_shape = (H_out, W_out, C, H_f, W_f)
    new_strides = (stride * s_h, stride * s_w, s_c, s_h, s_w)

    # Создаём виртуальную матрицу окон. ПАМЯТЬ ПРИ ЭТОМ НЕ КОПИРУЕТСЯ.
    x_windows = as_strided(x, shape=new_shape, strides=new_strides)

    # Теперь X_windows имеет форму (H_out, W_out, C, H_f, W_f)
    # W по-прежнему имеет форму (K, C, H_f, W_f)
    # Мы можем использовать np.tensordot для максимально быстрого перемножения
    # Суммируем по осям (C, HF, WF), которые являются последними тремя для X_windows
    # и последними тремя для W.

    out = np.tensordot(x_windows, W, axes=((2, 3, 4), (1, 2, 3)))
    return out.transpose(2, 0, 1)


def _conv2d_batch_ext(X: np.ndarray, W: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    [Расширенная двумерная свёртка для батча]
    Универсальная батч-свёртка через виртуальный 6D тензор окон.
    :param X: (N, C, H_x, W_x) - Батч входов.
    :param W: (K, C, H_f, W_f) - Расширенный фильтр (набор фильтров).
    :param stride: Страйд по пространственным осям (H, W).
    :return: (N, K, H_out, W_out) - Результат свёртки.
    """
    N, C, H_x, W_x = X.shape
    K, C_f, H_f, W_f = W.shape
    assert C == C_f, f"Каналы не совпадают: x={C}, W={C_f}"

    # 1. Расчет выходных размеров
    H_out = (H_x - H_f) // stride + 1
    W_out = (W_x - W_f) // stride + 1

    # 2. Получаем страйды исходного батча (N, C, H, W)
    # Для float32: s_w=4, s_h=W_x*4, s_c=H_x*W_x*4, s_n=C*H_x*W_x*4
    s_n, s_c, s_h, s_w = X.strides

    # 3. Строим "Тензор окон" (6D)
    # Форма: (Батч, H_сетки, W_сетки, Канал_окна, H_окна, W_окна)
    new_shape = (N, H_out, W_out, C, H_f, W_f)

    # Логика страйдов:
    # Прыжок по батчу (s_n), прыжки по сетке (stride*s_h, stride*s_w),
    # и стандартные прыжки внутри самого кубика (s_c, s_h, s_w).
    new_strides = (s_n, stride * s_h, stride * s_w, s_c, s_h, s_w)

    x_windows = as_strided(X, shape=new_shape, strides=new_strides)

    # 4. Вычисляем свертку через tensordot
    # Мы схлопываем (суммируем) последние 3 оси x_windows (C, H_f, W_f)
    # с осями фильтра W (C, H_f, W_f).
    # У x_windows это индексы (3, 4, 5). У W это индексы (1, 2, 3).
    res = np.tensordot(x_windows, W, axes=((3, 4, 5), (1, 2, 3)))

    # 5. Результат tensordot имеет форму (N, H_out, W_out, K)
    # Транспонируем в стандартный (N, K, H_out, W_out)
    return res.transpose(0, 3, 1, 2)


def _zero_pad(T: np.ndarray, n: int):
    if n == 0:
        return T
    N, K, H, W = T.shape
    zp = np.zeros((N, K, H + 2 * n, W + 2 * n))
    zp[:, :, n:H + n, n:W + n] = T
    return zp


def _dilate(T: np.ndarray, s: int):
    N, K, H, W = T.shape
    dil = np.zeros((N, K, s * (H - 1) + H, s * (W - 1) + W))
    dil[:, :, ::s + 1, ::s + 1] = T
    return dil


def _w2u(W: np.ndarray):
    """
    Поворачивает каждый фильтр вдоль оси глубины на pi,
    пересобирает новый расширенный фильтр, состоящий из фильтров,
    составленных из k-х матриц исходных фильтров.
    :param W: (K, C, H_f, W_f) - Расширенный фильтр.
    :return U: (C, K, H_f, W_f) Новый расширенный фильтр.
    """
    return np.flip(W, axis=(2, 3)).transpose(1, 0, 2, 3)


def _conv_test(
        z: np.ndarray,
        w: np.ndarray,
        s_n: int = 1,
        s_m: int = 1,
):
    """Свёртка циклом"""
    # filter size
    m = w.shape[-1]
    n = w.shape[-2]

    # output size
    m_inp = z.shape[-1]
    n_inp = z.shape[-2]
    m_out = np.floor((m_inp - m) / s_m + 1).astype(int)
    n_out = np.floor((n_inp - n) / s_n + 1).astype(int)

    # convolution
    out = np.zeros((n_out, m_out))
    for i in range(m_out):
        for j in range(n_out):
            out[i][j] = np.sum(np.multiply(z[:, i:i + m, j:j + n], w))
    return out


def _naive_conv2d_batch(X, W, stride=1):
    """Медленная эталонная реализация свёртки через циклы"""
    N, C, H_x, W_x = X.shape
    K, C_f, H_f, W_f = W.shape
    H_out = (H_x - H_f) // stride + 1
    W_out = (W_x - W_f) // stride + 1

    out = np.zeros((N, K, H_out, W_out))

    for n in range(N):
        for k in range(K):
            for i in range(H_out):
                for j in range(W_out):
                    # Вырезаем окно из входного тензора
                    window = X[n, :, i * stride: i * stride + H_f, j * stride: j * stride + W_f]
                    # Умножаем на k-й фильтр и суммируем все элементы (C, H_f, W_f)
                    out[n, k, i, j] = np.sum(window * W[k])
    return out


def _test_forward():
    N, C, H, W = 1, 3, 6, 6
    K, HF, WF = 5, 3, 3
    stride = 1
    X = np.random.randint(-3, 4, size=(N, C, H, W))
    W = np.random.randint(-1, 2, size=(K, C, HF, WF))
    b = np.array([i * 10 for i in range(K)])
    # print(b)
    B = b.reshape(1, K, 1, 1)
    # print(B)
    A = _conv2d_batch_ext(X, W, stride) + B
    print("Conv:")
    print(_conv2d_batch_ext(X, W, stride))
    print("B:")
    print(B)
    print("Activation:")
    print(A)


def _test_conv2d_batch():
    # 1. Параметры теста
    N, C, H, W = 2, 3, 10, 10  # Батч из 2 картинок 3x10x10
    K, HF, WF = 4, 3, 3  # 4 фильтра размером 3x3x3
    stride = 2

    # 2. Генерируем случайные данные
    # Используем float64 для максимальной точности сравнения
    x = np.random.randn(N, C, H, W).astype(np.float64)
    weights = np.random.randn(K, C, HF, WF).astype(np.float64)

    print(f"Запуск теста _conv2d_batch...")
    print(f"Вход: {x.shape}, Фильтры: {weights.shape}, Страйд: {stride}")

    # 3. Считаем обоими методами
    out_naive = _naive_conv2d_batch(x, weights, stride=stride)
    out_fast = _conv2d_batch_ext(x, weights, stride=stride)

    # 4. Сравниваем результаты
    # np.allclose проверяет, что разница между элементами ничтожно мала
    difference = np.max(np.abs(out_naive - out_fast))
    is_correct = np.allclose(out_naive, out_fast, atol=1e-10)

    print("-" * 30)
    print(f"Результат корректен: {is_correct}")
    print(f"Максимальное отклонение: {difference:.2e}")
    print(f"Форма выхода: {out_fast.shape}")

    if is_correct:
        print("✅ Тест пройден! Магия strides работает.")
    else:
        print("❌ Ошибка! Результаты не совпадают.")


if __name__ == '__main__':
    pass
