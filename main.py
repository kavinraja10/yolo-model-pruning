from ultralytics import YOLO
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from dataset import get_dataset,convert_coco_to_yolo
 


def prepare_dataset():
    get_dataset()
    convert_coco_to_yolo(r'data\annotations\instances_val2017.json', 'data')




if __name__ == "__main__":
    prepare_dataset()
    