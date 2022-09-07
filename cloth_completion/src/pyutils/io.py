"""Utilities for processing ORP data."""
import os

import numpy as np

# import cv2
from PIL import Image

# TODO: update this dependency
import torch
#import imageInterpolater

import logging
import cv2

cv2.setNumThreads(1)

logger = logging.getLogger()


# TODO: trying deterministic for comparison reasons
def seed_torch(seed=666):
    import random
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_obj(path):
    """Load wavefront OBJ from file."""
    v = []
    vt = []
    vindices = []
    vtindices = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append(
                    [int(entry.split("/")[0]) - 1 for entry in line.split()[1:]]
                )
                if line.find("/") != -1:
                    vtindices.append(
                        [int(entry.split("/")[1]) - 1 for entry in line.split()[1:]]
                    )

    return v, vt, vindices, vtindices


def write_obj(path, v, vt, vi, vti):
    """Write wavefront OBJ to file."""
    with open(path, "w") as f:
        for xyz in v:
            f.write("v {} {} {}\n".format(xyz[0], xyz[1], xyz[2]))
        for uv in vt:
            f.write("vt {} {}\n".format(uv[0], uv[1]))

        if len(vti) > 0:  # if UV is provided
            for i, ti in zip(vi, vti):
                f.write("f ")
                for vind, vtind in zip(i, ti):
                    f.write("{}/{} ".format(vind + 1, vtind + 1))
                f.write("\n")
        else:
            for i in vi:  # only vi
                f.write("f ")
                for vind in i:
                    f.write("{} ".format(vind + 1))
                f.write("\n")


def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters for Mugsy
    cameras."""
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrin": np.array(intrin),
                "dist": np.array(dist),
                "extrin": np.array(extrin),
            }

    return cameras


def load_ply(filename):
    """Loads a ply mesh.

    Args:
        filename: str, path to the file

    Returns:
        a tuple (verts, faces) of numpy arrays with dtype float32 and int32
    """
    # TODO: use different "backends" here?
    # import open3d as o3d
    # mesh = o3d.io.read_triangle_mesh(filename)
    # return (
    #     np.asarray(mesh.vertices, dtype=np.float32),
    #     np.asarray(mesh.triangles, dtype=np.int32),
    # )
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    return imageInterpolater.loadplyvet(filename)


def save_ply(filename, verts, faces=None):
    """Save a ply mesh.

    Args:
        filename: string, path to the filename
        verts: an array of vertices
        faces: an array of vertex indices

    """
    if verts.device.type != "cpu" or (faces is not None and faces.device.type != "cpu"):
        raise ValueError("only CPU-bound values are supported")
    if faces is not None:
        imageInterpolater.saveplymesh(filename, verts, faces)
    else:
        imageInterpolater.saveplyvet(filename, verts)


def load_calibration(path):
    """Returns: (names, intrin, extrin, dist)"""
    intrin = []
    extrin = []
    dist = []
    names = []
    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break
            intrin.append([[float(x) for x in f.readline().split()] for i in range(3)])
            dist.append([float(x) for x in f.readline().split()])
            extrin.append([[float(x) for x in f.readline().split()] for i in range(3)])
            names.append(name[:-1])
            f.readline()
    return names, np.array(intrin), np.array(extrin), np.array(dist)


def load_cameras(path, cids):
    all_cids, intrin, extrin, dist = load_calibration(path)
    idxs = [all_cids.index(cid) for cid in cids]
    return intrin[idxs], extrin[idxs], dist[idxs]


def downsample(image, factor):
    if factor == 1:
        return image
    import scipy.ndimage

    image = image.copy("C")
    for i in range(image.shape[-1]):
        image[..., i] = scipy.ndimage.filters.gaussian_filter(
            image[..., i], 2.0 * factor / 6
        )
    return image[::factor, ::factor]


def imread_exr(path):
    """Read an exr image."""
    # TODO: should this also be transposed?
    import cv2

    result = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if result is None:
        logger.warning(f"error when reading {path}")
    return result


def imread(path, dst_size=None, dtype=np.float32, interpolation=cv2.INTER_LINEAR):
    """Reads an image and resizes it to the given size.

    Args:
        path: str, path to the image
        dst_size: a tuple of ints, (H, W)
        dtype: the dtype of the resulting np.ndarray

    Returns:
        a numpy array with the image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    image = cv2.imread(path)  # , cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("error when reading the image")
    if dst_size and image.shape[:2] != dst_size:
        image = cv2.resize(image, dst_size[::-1], interpolation=interpolation)

    # transposing if necessary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
    return image.astype(np.float32)


def imread_pillow(path, dst_size=None, dtype=np.float32):
    """Reads an image and resizes it to the given size.

    Args:
        path: str, path to the image
        dst_size: a tuple of ints, (H, W)
        dtype: the dtype of the resulting np.ndarray

    Returns:
        a numpy array with the image
    """
    image = Image.open(path)
    if dst_size:
        image = image.resize(dst_size[::-1])
    image = np.array(image, dtype=dtype)
    # transposing if necessary
    if len(image.shape) == 3:
        image = image.transpose((2, 0, 1))
    return image


def imwrite(path, image):
    """Writing an image."""
    return Image.fromarray(image).save(path)


def imwrite_exr(path, image):
    import cv2

    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(f"cannot write to {path}, directory does not exist.")
    cv2.imwrite(path, image)
