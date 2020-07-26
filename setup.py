from setuptools import setup, find_packages

setup(
    name='yolo-v4',
    version='0.4',
    author='Philippe Remy',
    description='Interface Darknet YOLOv4 with python',
    include_package_data=True,
    data_files=[
    ],
    install_requires=[
        'numpy',
        'opencv-python',
        'scikit-image',
        'attrs',
        'Pillow'
    ],
    packages=find_packages(),
)
