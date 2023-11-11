import numpy as np
from scipy import special
from scipy import signal
import math


def positionencoding1D(W, L):

    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1

    x_el = []

    x_el_hf = []

    pe_1d = np.zeros((W, 2*L+1))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        x = np.sin(val * np.pi * x_linspace)
        x_el.append(x)

        x = np.cos(val * np.pi * x_linspace)
        x_el_hf.append(x)


    for x_i in range(0, W):

        p_enc = []

        for li in range(0, L):
            p_enc.append(x_el[li][x_i])
            p_enc.append(x_el_hf[li][x_i])

        p_enc.append(x_linspace[x_i])

        pe_1d[x_i] = np.array(p_enc)

    return pe_1d.astype('float32')



def positionencoding2D(W, H, L, basis_function,epoch):

    fre=np.clip(epoch/30000,0,1)
    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
    y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1

    x_el = []
    y_el = []

    x_el_hf = []
    y_el_hf = []

    pe_2d = np.zeros((W, H, 4*L+2))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        if basis_function == 'rbf':

            # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
            # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

            # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

            M_1 = np.random.rand(2, 2)

            phase_shift = np.random.rand(1) * np.pi

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
            x_el.append(x_1_y_1[0, :])
            y_el.append(x_1_y_1[1, :])

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
            x_el_hf.append(x_1_y_1[0, :])
            y_el_hf.append(x_1_y_1[1, :])

        elif basis_function == 'diric':

            x = special.diric(np.pi * x_linspace, val)
            x_el.append(x)

            x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
            x_el_hf.append(x)

            y = special.diric(np.pi * y_linspace, val)
            y_el.append(y)

            y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
            y_el_hf.append(y)

        elif basis_function == 'sawtooth':
            x = signal.sawtooth(val * np.pi * x_linspace)
            x_el.append(x)

            x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
            x_el_hf.append(x)

            y = signal.sawtooth(val * np.pi * y_linspace)
            y_el.append(y)

            y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
            y_el_hf.append(y)

        elif basis_function == 'sin_cos':

            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

    for y_i in range(0, H):
        for x_i in range(0, W):

            p_enc = []

            for li in range(0, L):
                p_enc.append(x_el[li][x_i])
                p_enc.append(x_el_hf[li][x_i])

                p_enc.append(y_el[li][y_i])
                p_enc.append(y_el_hf[li][y_i])

            p_enc.append(x_linspace[x_i])
            p_enc.append(y_linspace[y_i])

            pe_2d[x_i, y_i] = np.array(p_enc)

    return pe_2d.astype('float32')



def positionencoding3D(W, H, D, L1, L2):

    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
    y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1
    z_linspace = (np.linspace(0, D - 1, D) / D) * 2 - 1

    x_el = []
    y_el = []
    z_el = []

    x_el_hf = []
    y_el_hf = []
    z_el_hf = []

    pe_3d = np.zeros((W, H, D, 4*L1+3+2*L2))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L1):
        val = 2 ** el

        x = np.sin(val * np.pi * x_linspace)
        x_el.append(x)

        x = np.cos(val * np.pi * x_linspace)
        x_el_hf.append(x)

        y = np.sin(val * np.pi * y_linspace)
        y_el.append(y)

        y = np.cos(val * np.pi * y_linspace)
        y_el_hf.append(y)

        if el < L2:
            z = np.sin(val * np.pi * z_linspace)
            z_el.append(z)

            z = np.cos(val * np.pi * z_linspace)
            z_el_hf.append(z)


    for z_i in range(0, D):
        for y_i in range(0, H):
            for x_i in range(0, W):

                p_enc = []

                for li in range(0, L1):
                    p_enc.append(x_el[li][x_i])
                    p_enc.append(x_el_hf[li][x_i])

                    p_enc.append(y_el[li][y_i])
                    p_enc.append(y_el_hf[li][y_i])

                    if li < L2:
                        p_enc.append(z_el[li][z_i])
                        p_enc.append(z_el_hf[li][z_i])

                p_enc.append(x_linspace[x_i])
                p_enc.append(y_linspace[y_i])
                p_enc.append(z_linspace[z_i])

                pe_3d[x_i, y_i, z_i] = np.array(p_enc)

    return pe_3d.astype('float32')