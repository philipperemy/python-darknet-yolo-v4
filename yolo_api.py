import json
import logging
import os
import shutil
import subprocess
import xml.etree.cElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import attr
import imageio
import numpy as np
from PIL import Image
# noinspection PyUnresolvedReferences
from xml.dom import minidom

logger = logging.getLogger(__name__)


def get_script_arguments():
    parser = ArgumentParser('Skysense YOLO')
    parser.add_argument('--input', type=os.path.expanduser, required=True)
    parser.add_argument('--conf_path', default=None)
    parser.add_argument('--exclude', type=str, default=None)
    parser.add_argument('--model', choices=['v3', 'v4'], default='v4')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--silent', action='store_true')
    return parser.parse_args()


@attr.s
class Parameters:
    input = attr.ib(type=str)
    model = attr.ib(type=str)
    exclude = attr.ib(type=str, default=None)
    force = attr.ib(type=bool, default=False)
    gpu = attr.ib(type=bool, default=True)
    silent = attr.ib(type=bool, default=False)

    @classmethod
    def from_dict(cls, s: dict):
        return cls(
            input=s['input'],
            exclude=s['exclude'],
            model=s['model'],
            force=s['force'],
            gpu=s['gpu'],
            silent=s['silent']
        )


@attr.s
class DarkNetPredictionResult:
    # https://www.ccoderun.ca/programming/2019-08-18_Darknet_training_images/
    class_name = attr.ib(type=str)
    class_confidence = attr.ib(type=float)
    left_x = attr.ib(type=int)  # box center x.
    top_y = attr.ib(type=int)  # box center y.
    width = attr.ib(type=int)  # box width.
    height = attr.ib(type=int)  # box height.
    info = attr.ib(type=dict, default={})  # extra info about filename...


@attr.s
class DarkHelpCLIBuild:
    cfg_file = attr.ib(type=Path)
    weights_file = attr.ib(type=Path)
    names_file = attr.ib(type=Path)
    threshold_detection = attr.ib(type=float)
    mnt_volume = attr.ib(type=str)
    pred_dir = attr.ib(type=Path)
    txt_file = attr.ib(type=Path, default=None)
    images = attr.ib(type=list, default=[])

    def __str__(self):  # build CLI.
        assert len(self.images) > 0 or self.txt_file is not None
        cmd = f'/bin/bash -c "DarkHelp -j {self.cfg_file} {self.weights_file} {self.names_file} '
        if len(self.images) > 0:
            path_to_image = ' '.join([str(i) for i in self.images])
        else:
            path_to_image = f'-l {self.txt_file}'
        cmd += f'{path_to_image} -t {self.threshold_detection} -k;"'
        return cmd


def sort_prediction_results(p: List[DarkNetPredictionResult]) -> List[DarkNetPredictionResult]:
    return sorted(p, key=lambda x: (x.class_name, x.left_x, x.top_y, x.width, x.height, x.class_confidence))


def write_to_voc_xml_file(img_filename: Path, predictions: List[DarkNetPredictionResult]):
    img = Image.open(img_filename)
    w, h = img.size
    bdboxes_names = ['hemp'] * len(predictions)
    # create the file structure
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    filename = ET.SubElement(annotation, 'filename')
    path = ET.SubElement(annotation, 'path')
    source = ET.SubElement(annotation, 'source')
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    database = ET.SubElement(source, 'database')

    path.text = str(img_filename.resolve())
    folder.text = str(img_filename.parent)
    filename.text = str(img_filename.name)

    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')

    width.text = str(w)
    height.text = str(h)
    depth.text = str(3)  # depth value. RGB.

    assert len(bdboxes_names) == len(predictions)
    for box, bb_name in zip(predictions, bdboxes_names):
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = bb_name
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        pred = ET.SubElement(bndbox, 'pred')

        xmin1 = box.left_x
        xmax1 = box.left_x + box.width
        ymin1 = box.top_y
        ymax1 = box.top_y + box.height
        assert 0 <= xmin1 <= w
        assert 0 <= xmax1 <= w
        assert 0 <= ymin1 <= h
        assert 0 <= ymax1 <= h
        xmin.text = str(xmin1)
        ymin.text = str(ymin1)
        xmax.text = str(xmax1)
        ymax.text = str(ymax1)
        pred.text = str('true')

    # create a new XML file with the results
    with open(str(img_filename.with_suffix('.xml')), 'w') as w:
        w.write(minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   "))


class YOLO:

    def __init__(self,
                 cfg_file: Union[Path, str],
                 weights_file: Union[Path, str],
                 names_file: Union[Path, str],
                 cfg_data_file: Union[Path, str] = None):
        self.pwd = Path(__file__).parent.absolute()
        self.cfg_file = Path(cfg_file)
        self.weights_file = Path(weights_file)
        self.names_file = Path(names_file)
        self.cfg_data_file = Path(cfg_data_file)
        self.threshold_detection = 0.25
        self.local_pred_dir = self.pwd / 'predictions'
        config = open(self.cfg_file).read().strip().split('\n')
        self.w = int([w for w in config if w.startswith('width')][0].split('=')[1])
        self.h = int([w for w in config if w.startswith('height')][0].split('=')[1])

    def dark_help_inference(self, dir_or_images_list: Union[List[Path], Path, str]):
        if isinstance(dir_or_images_list, str) or isinstance(dir_or_images_list, Path):
            dir_or_images_list = Path(dir_or_images_list).expanduser()
            if Path(dir_or_images_list).exists() and Path(dir_or_images_list).is_dir():
                images_list = list(Path(dir_or_images_list).glob('**/*'))
            elif Path(dir_or_images_list).exists():
                images_list = [dir_or_images_list]
            else:
                raise FileNotFoundError()
        else:
            images_list = dir_or_images_list
        return self.dark_help_inference_list(images_list)

    def dark_help_inference_list(self, images_list: List[Path]):
        image_dir = self.pwd / 'img'
        for image in images_list:
            assert image.exists()
        if self.local_pred_dir.exists():
            shutil.rmtree(self.local_pred_dir)
        self.local_pred_dir.mkdir()
        if image_dir.exists():
            shutil.rmtree(image_dir)
        image_dir.mkdir()
        images_list = [img_path.expanduser() for img_path in images_list]
        # is it a big image? e.g. more than what the model expects?
        image_to_dir_map = {}
        for image_path in images_list:
            image_to_dir_map[image_path.name] = image_path.parent
            # copy as it is.
            if image_path.suffix.lower() == '.jpg':
                shutil.copy(str(image_path), str(image_dir))
            else:  # convert to JPG.
                # noinspection PyTypeChecker
                imageio.imwrite(image_dir / (image_path.stem + '.jpg'), np.array(Image.open(image_path)))
        # TODO: handle case like 320x320. as well.
        images_containers = [self.pwd / 'img' / p.name for p in image_dir.glob('*.jpg')]
        if len(images_list) > 1:  # command will become too long otherwise...
            local_txt_file = self.pwd / 'img.txt'
            with open(local_txt_file, 'w') as w:
                w.write('\n'.join([str(s) for s in images_containers]))
            dh = DarkHelpCLIBuild(
                cfg_file=self.cfg_file,
                weights_file=self.weights_file,
                names_file=self.names_file,
                txt_file=local_txt_file,
                threshold_detection=self.threshold_detection,
                mnt_volume=self.pwd,
                pred_dir=self.local_pred_dir
            )
        else:
            dh = DarkHelpCLIBuild(
                cfg_file=self.cfg_file,
                weights_file=self.weights_file,
                names_file=self.names_file,
                images=images_containers,
                threshold_detection=self.threshold_detection,
                mnt_volume=self.pwd,
                pred_dir=self.local_pred_dir
            )
        cmd = str(dh)
        output = json.loads(self.run_command(cmd).split('JSON OUTPUT')[1])

        # delete tmp contents.
        # directories = set()
        # for output_image in output['image']:
        #     directories.add(str(Path(output_image['annotated_image']).parent))
        # for directory in directories:
        #     print(directory)
        #     shutil.rmtree(directory)

        # DarkHelp to Dict[List[DarkNetPredictionResult]]
        results = {}
        for output_image in output['image']:
            if 'prediction' in output_image:
                local_filename = output_image['filename']
                real_filename = Path(image_to_dir_map[Path(local_filename).name]) / Path(local_filename).name
                assert Path(local_filename).exists()
                assert Path(real_filename).exists()
                output_image['filename'] = real_filename
                results[output_image['filename']] = []
                for prediction in output_image['prediction']:
                    results[output_image['filename']].append(DarkNetPredictionResult(
                        class_name=prediction['name'].rsplit(' ', 1)[0],
                        class_confidence=float(prediction['name'].rsplit(' ', 1)[-1].replace('%', '')),
                        left_x=prediction['rect']['x'],
                        top_y=prediction['rect']['y'],
                        width=prediction['rect']['width'],
                        height=prediction['rect']['height']
                        # info=output_image
                    ))

        results = {k: sort_prediction_results(v) for (k, v) in results.items()}
        for rr in sum(results.values(), []):
            print(rr)
        return results

    def run_command(self, cmd):
        # stderr=None prints too much.
        return subprocess.check_output(cmd, shell=True, stderr=open('/dev/null', 'wb')).decode('utf8')

    def darknet_inference(self, image: Path):  # one image at a time.
        cmd = "./darknet detector test "
        cmd += f"{self.cfg_data_file} {self.cfg_file} {self.weights_file} "
        cmd += f"-ext_output {image} -dont_show && "
        cmd += f"cp predictions.jpg predictions/{image.stem}-pred.jpg"
        darknet_result = self.run_command(cmd)
        results = []
        for prediction in [b for b in darknet_result.split('\n') if 'top_y' in b]:
            for char_to_replace in ['(', ')', ':', '\t', '%']:
                prediction = prediction.replace(char_to_replace, ' ')
            o = [s for s in prediction.split(' ') if s != '']
            # 'hemp: 27%	(left_x:   15   top_y:  138   width:   42   height:   34)'
            class_name, class_confidence, left_x_tag, left_x, top_y_tag, top_y, width_tag, width, height_tag, height = o
            assert abs(float(class_confidence)) < 100
            assert left_x_tag == 'left_x'
            assert top_y_tag == 'top_y'
            assert width_tag == 'width'
            assert height_tag == 'height'
            results.append(DarkNetPredictionResult(
                class_name=class_name,
                class_confidence=float(class_confidence),
                left_x=int(left_x),
                top_y=int(top_y),
                width=int(width),
                height=int(height)
            ))
        return sort_prediction_results(results)


def inference_batch_image(yolo: YOLO, image_list: List[Path]):
    all_predictions = yolo.dark_help_inference(image_list)
    for image_filename, predictions in all_predictions.items():
        write_to_voc_xml_file(image_filename, predictions)
    for image in image_list:
        if not image.with_suffix('.xml').exists():
            write_to_voc_xml_file(image, [])


def inference_cli(parameters: Parameters):
    if parameters.model == 'v3':
        cfg_file = 'cfg/yolov3-tiny.cfg'
        weights_file = 'backup/hemp-yolov3-tiny_best.weights'
    else:
        cfg_file = 'cfg/yolov4.cfg'
        weights_file = 'yolov4.weights'

    if not parameters.silent:
        logger.info('CONFIG:  ', cfg_file)
        logger.info('WEIGHTS: ', weights_file)
        logger.info('GPU:     ', parameters.gpu)
    yolo = YOLO(
        cfg_file=cfg_file,
        weights_file=weights_file,
        names_file='data/coco.names',
        cfg_data_file='cfg/coco.data',
    )
    img_or_dir = Path(parameters.input)
    assert img_or_dir.exists()

    images_list = []
    if img_or_dir.is_dir():
        for ext in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
            images_list.extend(img_or_dir.glob(f'**/*.{ext}'))
        if parameters.exclude is not None:
            images_list = list(set(images_list) - set(img_or_dir.glob('**/' + parameters.exclude)))
    else:
        images_list.append(img_or_dir)

    real_weights = Path(weights_file).resolve().name
    logger.info(f'Input: {len(images_list)} images, weights: {real_weights}.')
    inference_batch_image(yolo, images_list)


if __name__ == '__main__':
    inference_cli(Parameters.from_dict(vars(get_script_arguments())))
