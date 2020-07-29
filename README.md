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
d = Detector(gpu_id=0, lib_darknet_path='libdarknet.so')
img_arr = np.array(img.resize((d.network_width(), d.network_height())))
d.perform_detect(image_path_or_buf=img_arr, show_image=False)
```
