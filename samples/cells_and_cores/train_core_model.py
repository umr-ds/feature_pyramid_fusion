import os
import imgaug
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib


# Directory to save logs and trained model
MODEL_DIR = os.path.join("/data/logs/")


# Directory of images to run detection on
IMAGE_DIR = os.path.join("/data/")

from samples.cells_and_cores.cells_and_cores import CoresConfig

config = CoresConfig()
config.GPU_COUNT = 1
config.BATCH_SIZE = 2
config.IMAGE_MIN_DIM = 512
config.IMAGE_MAX_DIM = 512
config.IMAGES_PER_GPU = 2
config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
config.LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }



config.display()

# Create core model object
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR+"/cores", config=config)

# Load weights trained on MS-COCO
# model.load_core_weights("../logs/core_020180505T1505/mask_rcnn_core_0_0001.h5")

layernames = [ l.name for l in model.keras_model.layers ]
#print (len(layernames))
#print (layernames)
model.load_weights("/data/models/model.e120.hdf5", by_name=True,exclude=['avg_pool','fc1000'])

from samples.cells_and_cores.cells_and_cores import CoresDataset

# Training dataset
dataset_train = CoresDataset()
#dataset_train.load_data("/data/test","")
dataset_train.load_data("/data/","train")
dataset_train.prepare()

# Validation dataset
dataset_val = CoresDataset()
#dataset_val.load_data("/data/test","")
dataset_val.load_data("/data/","val")
dataset_val.prepare()

# Image Augmentation
# Right/Left flip 50% of the time
augmentation = imgaug.augmenters.Fliplr(0.5)


# Training - Stage 1
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads',
            augmentation=augmentation)

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=250,
            layers='4+',
            augmentation=augmentation)

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=350,
            layers='all',
            augmentation=augmentation)
