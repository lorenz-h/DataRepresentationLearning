import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

m = 28
n = 28
file_string = "photo.jpg"
image = scipy.misc.imread(file_string).astype(float)
image = image[..., 0]
dct = np.zeros_like(image)

dim_len = 28


def u(k, m):
    if u == 0:
        return np.sqrt(1/m)
    else:
        return np.sqrt(2/m)


def freq(k, r, m):
    f = (np.pi/m)*k*(r+0.5)
    return f


def C_kl(k, l, r, s):
    C = u(k, m)*u(l, n)*np.cos(freq(k, r, m))*np.cos(freq(l, s, n))
    return C


def C(r,s):
    C = np.zeros_like(image)
    for k in range(m):
        for l in range(n):
            C[k,l] = C_kl(k, l, r, s)
    return C


def T():
    tt = np.zeros([m, n, m, n])
    for r in range(m):
        for s in range(n):
            tt[r, s, ...] = C(r, s)
    return tt


plt.imshow(image, cmap="gray")
plt.show()
t_mat = T()
dct = np.tensordot(t_mat, image)
dct = np.tensordot(image, t_mat.T)
image = np.tensordot(t_mat.T, dct)
image = np.tensordot(image, t_mat)
plt.imshow(image, cmap="gray")
plt.show()
plt.imshow(dct, cmap="gray")
plt.show()