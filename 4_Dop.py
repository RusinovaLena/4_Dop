import math
import numpy as np
import random as rand


def gauss(matrix):
    m_len = len(matrix)
    b = np.zeros(m_len, dtype=float)
    for i in range(m_len):
        b[i] = 0
    n = m_len
    for k in range(n):
        my_max = k
        for i in range(k, n):
            if abs(matrix[i][k]) > abs(matrix[my_max][k]):
                my_max = i
        temp = matrix[k]
        matrix[k] = matrix[my_max]
        matrix[my_max] = temp
        t = b[k]
        b[k] = b[my_max]
        b[my_max] = t

        for i in range(k + 1, n):
            if matrix[k][k] == 0.0:
                continue
            else:
                a = matrix[i][k] / matrix[k][k]
                b[i] = b[i] - a * b[k]
                for j in range(k, n):
                    matrix[i][j] = matrix[i][j] - a * matrix[k][j]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, (-1), -1):
        sum = 0.0
        for j in range(i + 1, n):
            sum += matrix[i][j] * x[j]
        if abs(matrix[i][i]) < 0.0001:
            x[i] = 1
        else:
            x[i] = (b[i] - sum) / matrix[i][i]
    sum = 0.0
    for i in range(len(x)):
        sum = x[i] + sum
    for i in range(len(x)):
        if sum != 0.0:
            x[i] = x[i]/sum
        else:
            continue
    return x


def sub(matrix_1, matrix_2):
    m_0_len = len(matrix_1[0])
    m_len = len(matrix_1)
    sub_res = np.zeros((m_len, m_0_len), dtype=float)
    for i in range(m_len):
        for j in range(m_0_len):
            sub_res[i][j] = matrix_1[i][j] - matrix_2[i][j]
    return sub_res


def factorial(n):
    f = 1
    for i in range(1, n+1):
        f *= i
    return f


def transpose(matrix):
    m_0_len = len(matrix[0])
    m_len = len(matrix)
    t_mtr = np.zeros((m_0_len, m_len), dtype=float)
    for i in range(m_len):
        for j in range(m_0_len):
            t_mtr[j][i] = matrix[i][j]
    return t_mtr


def new_msg(lamb):
    l = float(math.exp(-lamb))
    p = 1.0
    k = 0
    while True:
        k = k + 1
        p *= rand.uniform(0, 1)
        if not p > l:
            break
    return k - 1


def theory():
    bufSize = 3
    b = bufSize + 1
    k = 1
    for l in np.arange(0, 3, 0.1):
        matrix = np.zeros((b, b), dtype=float)
        matrix[0][0] = math.exp(-l)
        for i in range(b):
            if i != b-1:
                matrix[0][i] = math.exp(-l) * (math.pow(l, i)/factorial(i))
                if i != 0:
                    matrix[i][i-1] = math.exp(-l)
                    matrix[i][i] = l * math.exp(-l)
                    if i + k < b - 1:
                        matrix[i][i+k] = (math.pow(l, k+1)/factorial(k + 1)) * (math.exp(-l))
                        k = k + 1
        for i in range(b):
            sum = 0.0
            for j in range(b):
                sum = sum + matrix[i][j]
            if i == 0:
                matrix[i][b-1] = matrix[i][b-1] + (1 - sum)
            else:
                matrix[i][b-2] = matrix[i][b-2] + (1 - sum)
        I = np.zeros((b, b), dtype=float)
        for i in range(b):
            I[i][i] = 1
        x = gauss(sub(transpose(matrix), I))
        E_N = 0
        for i in range(1, b):
            E_N = E_N + x[i] * i
        output_l = 1 - x[0]
        if output_l == 0:
            continue
        else:
            E_D = E_N / output_l
        print("lambda:", "%.1f" % l, "\t", "E[N]: ", E_N, "\t", "E[D]: ", E_D, "\t", "output lamda: ", output_l)


def modeling():
    for l in np.arange(0.0, 3.0, 0.1):
        frames = 100000
        buffer_size = 3
        current_b_s = 0
        messages = 0
        sent_messages = 0
        for i in range(frames):
            newMessages = new_msg(float(l))
            if (current_b_s + newMessages) >= (buffer_size - 1):
                newMessages = buffer_size - 1 - current_b_s
                current_b_s = buffer_size - 1
            else:
                current_b_s = current_b_s + newMessages
            messages = messages + newMessages
            if current_b_s > 0:
                sent_messages = sent_messages + 1
                current_b_s = current_b_s - 1

        E_N = float(messages / frames)
        output_l = float(sent_messages / frames)
        if output_l == 0:
            continue
        else:
            E_D = E_N / output_l
        print("lambda:", "%.1f" % l, "\t", "E[N]: ", E_N, "\t", "E[D]: ", E_D, "\t", "output lamda: ", output_l)


work = 1
if work == 1:
    theory()
else:
    modeling()
