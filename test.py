from threading import Thread
from time import time

import cv2
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


def batch_single():
    img_samples = ['data/person.jpg', 'data/dog.jpg', 'data/person.jpg',
                   'data/person.jpg', 'data/dog.jpg', 'data/person.jpg']
    bs = len(img_samples)
    image_list = [cv2.imread(k) for k in img_samples]
    img_list = []
    d = Detector(gpu_id=0, lib_darknet_path='lib/libdarknet.so', batch_size=bs)
    for custom_image_bgr in image_list:
        custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        custom_image = cv2.resize(
            custom_image, (d.network_width(), d.network_height()), interpolation=cv2.INTER_NEAREST)
        custom_image = custom_image.transpose(2, 0, 1)
        img_list.append(custom_image)
    for _ in tqdm(range(900)):
        d.perform_batch_detect(img_list, batch_size=bs)


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
    batch_single()
