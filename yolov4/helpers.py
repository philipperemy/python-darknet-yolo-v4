import os
from ctypes import *
from pathlib import Path

import attr
import numpy as np


@attr.s
class DarkNetPredictionResult:
    class_name = attr.ib(type=str)  # name of the class detected. E.g. dog.
    class_confidence = attr.ib(type=float)  # probability of the class. E.g. 95%.
    left_x = attr.ib(type=int)  # box center x.
    top_y = attr.ib(type=int)  # box center y.
    width = attr.ib(type=int)  # box width.
    height = attr.ib(type=int)  # box height.
    info = attr.ib(type=dict, default={})  # extra info. Can be blank.


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def init_lib(lib_darknet_path):
    # lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
    # lib = CDLL("libdarknet.so", RTLD_GLOBAL)
    hasGPU = True
    if os.name == "nt":
        cwd = os.path.dirname(__file__)
        os.environ['PATH'] = cwd + ';' + os.environ['PATH']
        winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
        winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
        envKeys = list()
        for k, v in os.environ.items():
            envKeys.append(k)
        try:
            try:
                tmp = os.environ["FORCE_CPU"].lower()
                if tmp in ["1", "true", "yes", "on"]:
                    raise ValueError("ForceCPU")
                else:
                    print("Flag value '" + tmp + "' not forcing CPU mode")
            except KeyError:
                # We never set the flag
                if 'CUDA_VISIBLE_DEVICES' in envKeys:
                    if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                        raise ValueError("ForceCPU")
                try:
                    global DARKNET_FORCE_CPU
                    if DARKNET_FORCE_CPU:
                        raise ValueError("ForceCPU")
                except NameError:
                    pass
                # print(os.environ.keys())
                # print("FORCE_CPU flag undefined, proceeding with GPU")
            if not os.path.exists(winGPUdll):
                raise ValueError("NoDLL")
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
        except (KeyError, ValueError):
            hasGPU = False
            if os.path.exists(winNoGPUdll):
                lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
                print("Notice: CPU-only mode")
            else:
                # Try the other way, in case no_gpu was
                # compile but not renamed
                lib = CDLL(winGPUdll, RTLD_GLOBAL)
                print(
                    "Environment variables indicated a CPU run, but we didn't find `" + winNoGPUdll + "`. Trying a GPU run anyway.")
    else:
        lib = CDLL(Path(lib_darknet_path).resolve(), RTLD_GLOBAL)

    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int
    copy_image_from_bytes = lib.copy_image_from_bytes
    copy_image_from_bytes.argtypes = [IMAGE, c_char_p]
    return lib, hasGPU


def read_alt_names(meta_path: str):
    try:
        with open(meta_path) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
                        return altNames
            except TypeError:
                pass
    except Exception:
        pass
    return None
