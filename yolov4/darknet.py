import os
import random
from concurrent.futures.thread import ThreadPoolExecutor
from ctypes import *
from pathlib import Path
from threading import Lock
from typing import List

from yolov4.helpers import init_lib, DETECTION, DETNUMPAIR, METADATA, IMAGE, read_alt_names, DarkNetPredictionResult


class Detector:

    def __init__(
            self,
            config_path='cfg/yolov4.cfg',
            weights_path='yolov4.weights',
            meta_path='cfg/coco.data',
            lib_darknet_path='libdarknet.so',
            gpu_id=None
    ):
        """
        :param config_path: Path to the configuration file. Raises ValueError if not found.
        :param weights_path: Path to the weights file. Raises ValueError if not found.
        :param meta_path: Path to the data file. Raises ValueError if not found.
        :param lib_darknet_path: Path to the darknet library (.so in linux).
        :param gpu_id: GPU on which to perform the inference.
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.meta_path = meta_path
        self.gpu_id = gpu_id
        # to make sure we have only one inference per GPU.
        self.lock = Lock()

        self.net_main = None
        self.meta_main = None
        self.alt_names = None

        self.lib, self.has_gpu = init_lib(lib_darknet_path)

        self.predict = self.lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        # definition of all the bindings (functions to interface with C).
        if self.has_gpu:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        self.init_cpu = self.lib.init_cpu

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int),
                                           c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_batch_detections = self.lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        self.predict_image_letterbox = self.lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        self.network_predict_batch = self.lib.network_predict_batch
        self.network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                               c_float, c_float, POINTER(c_int), c_int, c_int]
        self.network_predict_batch.restype = POINTER(DETNUMPAIR)

        if not Path(self.config_path).exists():
            raise ValueError('Invalid config path `' + os.path.abspath(self.config_path) + '`')
        if not Path(self.weights_path).exists():
            raise ValueError('Invalid weight path `' + os.path.abspath(self.weights_path) + '`')
        if not Path(meta_path).exists():
            raise ValueError('Invalid data file path `' + os.path.abspath(self.meta_path) + '`')
        if self.gpu_id is not None:
            print(f'GPU -> {self.gpu_id}.')
            self.set_gpu(self.gpu_id)
        # batch size = 1
        self.net_main = self.load_net_custom(self.config_path.encode('ascii'), self.weights_path.encode('ascii'), 0, 1)
        self.meta_main = self.load_meta(self.meta_path.encode('ascii'))
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        self.alt_names = read_alt_names(self.meta_path)

        self.darknet_image = self.make_image(self.network_width(), self.network_height(), 3)

    def network_width(self):
        net = self.net_main
        return self.lib.network_width(net)

    def network_height(self):
        net = self.net_main
        return self.lib.network_height(net)

    def classify(self, im):
        net, meta = self.net_main, self.meta_main
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            if self.alt_names is None:
                nameTag = meta.names[i]
            else:
                nameTag = self.alt_names[i]
            res.append((nameTag, out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        #
        # custom_image_bgr = cv2.imread(image.decode('utf8'))  # use: detect(,,imagePath,)
        # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        # custom_image = cv2.resize(custom_image, (self.network_width(), self.network_height()),
        #                           interpolation=cv2.INTER_LINEAR)
        # img = Image.open(image.decode('utf8'))
        # custom_image2 = np.array(img.resize((self.network_width(), self.network_height())))
        # self.copy_image_from_bytes(self.darknet_image, custom_image2.tobytes())
        # im2 = self.load_image(image, 0, 0)
        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError('Invalid image path `' + os.path.abspath(image) + '`')
            image = image.encode('ascii')
            im = self.load_image(image, 0, 0)
        else:
            self.copy_image_from_bytes(self.darknet_image, image.tobytes())
            im = self.darknet_image

        ret = self.detect_image(im, thresh, hier_thresh, nms, debug)
        if isinstance(image, str):
            self.free_image(im)  # just when we load a new image.
        return ret

    def detect_image(self, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        net, meta = self.net_main, self.meta_main
        num = c_int(0)
        if debug:
            print('Assigned num')
        pnum = pointer(num)
        if debug:
            print('Assigned pnum')
        self.predict_image(net, im)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        if debug:
            print('did prediction')
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        if debug:
            print('Got dets')
        num = pnum[0]
        if debug:
            print('got zeroth index of pnum')
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        if debug:
            print('did sort')
        res = []
        if debug:
            print('about to range')
        for j in range(num):
            if debug:
                print('Ranging on ' + str(j) + ' of ' + str(num))
            if debug:
                print('Classes: ' + str(meta), meta.classes, meta.names)
            for i in range(meta.classes):
                if debug:
                    print('Class-ranging on ' + str(i) + ' of ' + str(meta.classes) + '= ' + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.alt_names is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.alt_names[i]
                    if debug:
                        print('Got bbox', b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug:
            print('did range')
        res = sorted(res, key=lambda x: -x[1])
        if debug:
            print('did sort')
        self.free_detections(dets, num)
        if debug:
            print('freed detections')
        return res

    def perform_detect(
            self,
            image_path_or_buf='data/dog.jpg',
            thresh: float = 0.25,
            show_image: bool = True,
            make_image_only: bool = False,
    ):
        self.lock.acquire()
        assert 0 < thresh < 1, 'Threshold should be a float between zero and one (non-inclusive)'
        detections = self.detect(image_path_or_buf, thresh)
        if show_image and isinstance(image_path_or_buf, str):
            try:
                from skimage import io, draw
                import numpy as np
                image = io.imread(image_path_or_buf)
                print('*** ' + str(len(detections)) + ' Results, color coded by confidence ***')
                imcaption = []
                for detection in detections:
                    label = detection[0]
                    confidence = detection[1]
                    pstring = label + ': ' + str(np.rint(100 * confidence)) + '%'
                    imcaption.append(pstring)
                    print(pstring)
                    bounds = detection[2]
                    shape = image.shape
                    # x = shape[1]
                    # xExtent = int(x * bounds[2] / 100)
                    # y = shape[0]
                    # yExtent = int(y * bounds[3] / 100)
                    yExtent = int(bounds[3])
                    xEntent = int(bounds[2])
                    # Coordinates are around the center
                    xCoord = int(bounds[0] - bounds[2] / 2)
                    yCoord = int(bounds[1] - bounds[3] / 2)
                    boundingBox = [
                        [xCoord, yCoord],
                        [xCoord, yCoord + yExtent],
                        [xCoord + xEntent, yCoord + yExtent],
                        [xCoord + xEntent, yCoord]
                    ]
                    # Wiggle it around to make a 3px border
                    rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox],
                                                    shape=shape)
                    rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                      shape=shape)
                    rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                      shape=shape)
                    rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox],
                                                      shape=shape)
                    rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox],
                                                      shape=shape)
                    boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                    draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
                    draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
                    draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
                    draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
                    draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)
                if not make_image_only:
                    io.imshow(image)
                    io.show()
                detections = {
                    'detections': detections,
                    'image': image,
                    'caption': '\n<br/>'.join(imcaption)
                }
            except Exception as e:
                print('Unable to show image: ' + str(e))

        results = []
        sub_detections = detections['detections'] if 'detections' in detections else detections
        for detection in sub_detections:
            class_name, class_confidence, bbox = detection
            x, y, w, h = bbox
            x_min = int(x - (w / 2))
            y_min = int(y - (h / 2))
            results.append(DarkNetPredictionResult(
                class_name=class_name,
                class_confidence=class_confidence,
                left_x=max(0, int(x_min)),
                top_y=max(0, int(y_min)),
                width=max(0, int(w)),
                height=max(0, int(h))
            ))
        self.lock.release()
        return results


class MultiGPU:

    def __init__(self, detectors_list: List[Detector]):
        self.detectors = {}
        for detector in detectors_list:
            gpu_id = detector.gpu_id
            if gpu_id is None:
                raise ValueError('To use MultiGPU, every gpu_id should be defined.')
            self.detectors[gpu_id] = detector
        self.num_gpus = len(self.detectors)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_gpus, thread_name_prefix='detector_')
        self.counter = 0  # used to dispatch, assuming the load is same on each GPU.
        self.locks = {}
        for gpu_id in self.detectors.keys():
            self.locks[gpu_id] = Lock()

    def find_available_gpu(self):
        list_to_choose_from = []
        for gpu_id, lock in self.locks.items():
            if not lock.locked():
                list_to_choose_from.append(gpu_id)
        if len(list_to_choose_from) == 0:
            list_to_choose_from = list(self.locks.keys())
        return random.choice(list_to_choose_from)

    def _perform_detect(self, gpu_id, *args):
        return self.detectors[gpu_id].perform_detect(*args)

    def perform_detect(self, *args, **kwargs):
        gpu_id = self.find_available_gpu()
        lock = self.locks[gpu_id]
        with lock:
            # print(f'Lock for GPU {gpu_id} acquired.')

            def run(*arg, **kwarg):
                return self.detectors[gpu_id].perform_detect(*arg, **kwarg)

            future = self.thread_pool.submit(fn=run, *args, **kwargs)
            return future.result()
