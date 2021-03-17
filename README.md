## YOLOv4 in Python

Python interface to Darknet Yolo V4. The multi GPU is supported (load balancer).

### Installation

Compile the Darknet framework first.

```bash
sudo apt-get update 
sudo apt-get install -y pkg-config git build-essential libopencv-dev wget cmake
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make LIBSO=1 OPENCV=1 GPU=1 AVX=1 OPENMP=1 CUDNN=1 CUDNN_HALF=1 OPENMP=1 -j $(nproc)
chmod +x darknet
```

Then, download the weights by following the instructions here: https://github.com/AlexeyAB/darknet.

From there, create a virtual environment with python3.6+ and run this command:

```bash
pip install yolo-v4
```

### Run inference on images

To run inference on the GPU on an image `data/dog.jpg`, run this script:

```python
import numpy as np
from PIL import Image

from yolov4 import Detector

img = Image.open('data/dog.jpg')
d = Detector(gpu_id=0)
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

<p align="center">
  <img src="misc/sample1.png" width="500">
</p>

