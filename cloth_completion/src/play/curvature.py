import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.ndimage as  ndi
from torch.nn.functional import conv3d

# size of kernel
K = 7


def gaussian(x, sigma, mu=None):
    if mu is None:
        mu = np.zeros(len(x))

    return 1 / (sigma * np.sqrt(2 * np.pi)) \
           * np.exp(-sum((x_ - mu_) ** 2
                         for x_, mu_
                         in zip(x, mu)) \
                    / (2 * sigma ** 2))


def get_first_order(sigma=2, k=K, n=3):
    # 3D gaussian derivatives
    # returns Gx Gy Gz
    x = np.ogrid[[slice(-k, k + 1)] * n]
    g = gaussian(x, sigma=sigma)
    return np.stack([-x_ * g for x_ in x]) / sigma ** 2


def get_second_order(sigma=2, k=K, n=3):
    # 3D gaussian derivatives
    # Gxx, Gxy, Gxz, Gyy, Gyz, Gzz
    # Gxy = Gyx, Gxz = Gzx, Gyz = Gzy
    x = np.ogrid[[slice(-k, k + 1)] * n]
    g = gaussian(x, sigma=sigma)
    return np.stack([xi * xj * g if i != j else (xi - sigma) * (xj + sigma) * g
                     for j, xj in enumerate(x)
                     for i, xi in enumerate(x)
                     if i >= j]) / sigma ** 4


def get_weights(convolve=False, as_tensor=True, volatile=True, **kwargs):
    # Gx, Gy, Gz, Gxx, Gxy, Gxz, Gyy, Gyz, Gzz
    weights = np.expand_dims(np.vstack(
        (get_first_order(**kwargs),
         get_second_order(**kwargs))), 1).astype('float32')
    if as_tensor:
        if convolve:
            weights = weights[:, :, ::-1, ::-1, ::-1]  # for true convolution in pytorch
        return torch.autograd.Variable(torch.from_numpy(weights.copy()),
                                       volatile=volatile)
    else:
        return weights


def get_data(fp, as_tensor=True, volatile=True):
    if isinstance(fp, np.ndarray):
        data = fp.astype('float32')
    else:
        data = np.load(fp).astype('float32')
    if as_tensor:
        data_tensor = torch.autograd.Variable(
            torch.from_numpy(
                data[None, None, ...]),
            volatile=volatile)
        return data_tensor
    else:
        return data


def norm2(y):
    return (y[:, :3] ** 2).sum(dim=1)


def mean_curvature(y, n2, ep=1e-7):
    '''
    3D only
    expects torch tensor (1,9,1,:,:,:)
    0  1  2  3   4   5   6   7   8
    Fx Fy Fz Fxx Fxy Fxz Fyy Fyz Fzz

    Mean curvature =

    -0.5 div(grad(F)/|grad(F)|)
    '''
    p = y[:, 0] ** 2 * (y[:, 6] + y[:, 8]) \
        + y[:, 1] ** 2 * (y[:, 3] + y[:, 8]) \
        + y[:, 2] ** 2 * (y[:, 3] + y[:, 6]) \
        - 2 * y[:, 0] * y[:, 1] * y[:, 4] \
        - 2 * y[:, 0] * y[:, 2] * y[:, 5] \
        - 2 * y[:, 1] * y[:, 2] * y[:, 7]
    return -0.5 * p / (n2 ** (3 / 2) + ep)


def gaussian_curvature(y, n2, ep=1e-7):
    '''
    3D only
    expects torch tensor (1,9,1,:,:,:)
    Fx Fy Fz Fxx Fxy Fxz Fyy Fyz Fzz

    Gaussian curvature =

      | Fxx Fxy Fxz Fx |
      | Fxy Fyy Fyz Fy |
    - | Fxz Fyz Fzz Fz |
      | Fx  Fy  Fz  0  | / || grad(F) ||**4

      -Fx**2*Fyy*Fzz + Fx**2*Fyz**2 + 2*Fx*Fxy*Fy*Fzz
      - 2*Fx*Fxy*Fyz*Fz - 2*Fx*Fxz*Fy*Fyz + 2*Fx*Fxz*Fyy*Fz
      - Fxx*Fy**2*Fzz + 2*Fxx*Fy*Fyz*Fz - Fxx*Fyy*Fz**2
      + Fxy**2*Fz**2 - 2*Fxy*Fxz*Fy*Fz + Fxz**2*Fy**2

    '''

    det = -y[:, 0] ** 2 * y[:, 6] * y[:, 8] + y[:, 0] ** 2 * y[:, 7] ** 2 \
          + 2 * y[:, 0] * y[:, 4] * y[:, 1] * y[:, 8] - 2 * y[:, 0] * y[:, 4] * y[:, 7] * y[:, 2] \
          - 2 * y[:, 0] * y[:, 5] * y[:, 1] * y[:, 7] + 2 * y[:, 0] * y[:, 5] * y[:, 6] * y[:, 2] \
          - y[:, 3] * y[:, 1] ** 2 * y[:, 8] + 2 * y[:, 3] * y[:, 1] * y[:, 7] * y[:, 2] \
          - y[:, 3] * y[:, 6] * y[:, 2] ** 2 + y[:, 4] ** 2 * y[:, 2] ** 2 \
          - 2 * y[:, 4] * y[:, 5] * y[:, 1] * y[:, 2] + y[:, 5] ** 2 * y[:, 1] ** 2
    return -det / (n2 ** 2 + ep)


def shape_index(H, K, ep=1e-7):
    # simplified from Eq 1, http://openaccess.city.ac.uk/4386/
    # For some reason need to invert the sign, original equation was this:
    # return 0.5 - torch.atan2(H.data,(H.data**2 - K.data).sqrt())/np.pi
    return 0.5 + torch.atan2(H.data, (H.data ** 2 - K.data).sqrt() + ep) / np.pi


if __name__ == '__main__':
    # luna data
    # fp = '/data/luna16_preproc/subset0/1011.1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273.npy'
    # data,mask = np.load(fp)[:,0]
    # sl = mask.max(axis=0).max(axis=0).argmax()
    # noisy sphere
    xyz = np.ogrid[-64:64, -64:64, -64:64]
    sphere = (sum(_ ** 2 for _ in xyz) < 1024).astype('float32')
    cylinder = (sum(_ ** 2 for _ in xyz[:2]) + np.zeros_like(xyz[2]) < 1024).astype('float32')
    data = sphere  # should be close to 1, cylinder close to 0.75
    data += np.random.rand(128, 128, 128) * 0.1
    sl = 64

    d = get_data(data)
    w = get_weights()

    # Fx Fy Fz Fxx Fxy Fxz Fyy Fyz Fzz
    y = conv3d(d.cuda(), w.cuda(), bias=None, padding=K).cpu()

    # Fx**2 + Fy**2 + Fz**2
    n2 = norm2(y)

    H = mean_curvature(y, n2)
    K = gaussian_curvature(y, n2)
    SI = shape_index(H, K)

    output = y.data.numpy().squeeze()
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    for d, ax in zip(output, axes.flatten()):
        ax.imshow(d[..., sl], cmap=plt.get_cmap('gray'))
        ax.set_xticks(())
        ax.set_yticks(())
    for d, ax in zip((H.data, K.data, SI), axes.flatten()[-3:]):
        ax.imshow(d.numpy().squeeze()[..., sl], vmin=-1, vmax=1, cmap=plt.get_cmap('gray'))
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()