## YOLOv4 in Python

Using python to interface with Darknet Yolo V4

```
pip install yolo-v4
```

Compile the `darknet` framework https://github.com/AlexeyAB/darknet by setting `LIBSO=1` in the Makefile and do `make`.


```python
import numpy as np
from PIL import Image

from yolov4 import Detector

img = Image.open('data/dog.jpg')
d = Detector(gpu_id=0, lib_darknet_path='lib/libdarknet.so')
img_arr = np.array(img.resize((d.network_width(), d.network_height())))
detections = d.perform_detect(image_path_or_buf=img_arr, show_image=False)
for detection in detections:
    box = detection.left_x, detection.top_y, detection.width, detection.height
    print(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % | {box}')
```

```c
dog          | 97.6 % | (100, 236, 147, 334)
truck        | 93.0 % | (367, 81, 175, 98)
bicycle      | 92.0 % | (90, 134, 362, 315)
pottedplant  | 34.1 % | (538, 115, 29, 47)
```
