from threading import Thread
from time import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolov4 import Detector, MultiGPU

c = 0


def target(g, desc):
    global c
    for _ in range(1000):
        # print(desc)
        g.perform_detect(show_image=False)
        if c % 100 == 0:
            print(time(), c)
        c += 1
    print(time(), c)


def main_single():
    img = Image.open('data/dog.jpg')
    d = Detector(gpu_id=0, lib_darknet_path='lib/libdarknet.so')
    img_arr = np.array(img.resize((d.network_width(), d.network_height())))
    for _ in tqdm(range(900)):
        d.perform_detect(image_path_or_buf=img_arr, show_image=False)


def main():
    g = MultiGPU([
        Detector(gpu_id=0, lib_darknet_path='lib/libdarknet.so'),
        Detector(gpu_id=1, lib_darknet_path='lib/libdarknet.so')
    ])

    thread1 = Thread(target=target, args=(g, 't1'))
    thread2 = Thread(target=target, args=(g, 't2'))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()


if __name__ == '__main__':
    main_single()
