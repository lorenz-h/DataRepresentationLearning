import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

m = 28
n = 28
file_string = "photo.jpg"
image = scipy.misc.imread(file_string).astype(float)
image = image[..., 0]
dct = np.zeros_like(image)


def u_k(k, m):
    if k == 0:
        return np.sqrt(1/m)
    else:
        return np.sqrt(2/m)


def v_l(l, n):
    if l == 0:
        return np.sqrt(1/n)
    else:
        return np.sqrt(2/n)


def freq(k, r, m):
    f = (np.pi/m)*k*(r+0.5)
    return f


def a_kl(k, l):
    summe = 0
    for r in range(0, m):
        for s in range(0, n):
            summe += image[r, s]*np.cos(freq(k, r, m))*np.cos(freq(l, s, n))

    return u_k(k, m)*v_l(l, n) * summe


coeffs = []

for k in range(0, m):
    for l in range(0, n):
        dct[k, l] = a_kl(k, l)
        coeffs.append((a_kl(k, l), k*l))
coeffs = sorted(coeffs, key=lambda x: x[1])
coeffs = [i[0] for i in coeffs]  # remove k*l entry
coeffs = np.array(coeffs)
coeffs = np.reshape(coeffs, [28, 28])
plt.imshow(dct, cmap="gray")
plt.show()
plt.imshow(coeffs, cmap="gray")
plt.show()
