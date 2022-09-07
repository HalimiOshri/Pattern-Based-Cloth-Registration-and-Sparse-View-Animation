import sys
import torch
import numpy as np
import struct

def ReadTensorFromBinaryFile(inputFile):
    f = open(inputFile,"rb")
    dtype = None
    dtypestr =  struct.unpack('i', f.read(4))[0]
    if dtypestr == 0:
        dtype = np.int32
    elif dtypestr == 1:
        dtype = np.float32
    elif dtypestr == 2:
        dtype = np.int64
    elif dtypestr == 3:
        dtype = np.float64
    else:
        print("ERROR: Not supported type %d" % dtypestr)

    ndims = struct.unpack('i', f.read(4))[0]
    dims = struct.unpack('i' * ndims, f.read(4 * ndims))
    nelem = np.prod(dims)
    np_array = np.fromfile(f, dtype=dtype, count=nelem).reshape(dims)
    return torch.tensor(np_array)

def WriteTensorToBinaryFile(inputTensor,outputFile):
    arr = inputTensor.detach().numpy()
    f = open(outputFile,"wb")
    if arr.dtype == np.int32:
        f.write(struct.pack('i',0))
    elif arr.dtype == np.float32:
        f.write(struct.pack('i',1))
    elif arr.dtype == np.int64:
        f.write(struct.pack('i',2))
    elif arr.dtype == np.float64:
        f.write(struct.pack('i',3))
    else:
        print("ERROR: Not supported type:")
        print(arr.dtype)

    f.write(struct.pack('i', arr.ndim))
    for i in range(arr.ndim):
        f.write(struct.pack('i', arr.shape[i]))
    arr.tofile(f)
    f.close()

def ReadArrayFromBinaryFile(inputFile):
    f = open(inputFile,"rb")
    dtype = None
    dtypestr =  struct.unpack('i', f.read(4))[0]
    if dtypestr == 0:
        dtype = np.int32
    elif dtypestr == 1:
        dtype = np.float32
    elif dtypestr == 2:
        dtype = np.int64
    elif dtypestr == 3:
        dtype = np.float64
    else:
        print("ERROR: Not supported type %d" % dtypestr)

    ndims = struct.unpack('i', f.read(4))[0]
    dims = struct.unpack('i' * ndims, f.read(4 * ndims))
    nelem = np.prod(dims)
    np_array = np.fromfile(f, dtype=dtype, count=nelem).reshape(dims)
    return np_array

def WriteArrayToBinaryFile(inputTensor,outputFile):
    arr = inputTensor
    f = open(outputFile,"wb")
    if arr.dtype == np.int32:
        f.write(struct.pack('i',0))
    elif arr.dtype == np.float32:
        f.write(struct.pack('i',1))
    elif arr.dtype == np.int64:
        f.write(struct.pack('i',2))
    elif arr.dtype == np.float64:
        f.write(struct.pack('i',3))
    else:
        print("ERROR: Not supported type")
        print(arr.dtype)
    
    f.write(struct.pack('i', arr.ndim))
    for i in range(arr.ndim):
        f.write(struct.pack('i', arr.shape[i]))
    arr.tofile(f)
    f.close()
