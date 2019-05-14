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

from samples.cells_and_cores.cells_and_cores import Cells2ChannelConfig

config = Cells2ChannelConfig()
config.GPU_COUNT = 1
config.BATCH_SIZE = 2
config.IMAGE_MIN_DIM = 512
config.IMAGE_MAX_DIM = 512
config.IMAGES_PER_GPU = 2
config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
config.LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 2.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 2.,
        "mrcnn_mask_loss": 2.
    }



config.display()

# Create core model object
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR+"/cells_2channel", config=config)

layernames = [ l.name for l in model.keras_model.layers ]

model.load_weights("/data/models/2_chan_model.e120.hdf5", by_name=True,exclude=['avg_pool','fc1000'])

from samples.cells_and_cores.cells_and_cores import  Cells2ChannelDataset

# Training dataset
dataset_train = Cells2ChannelDataset()
dataset_train.FIXED_INPUT_SHAPE = True
dataset_train.INPUT_SHAPE = [512, 512]
#dataset_train.load_data("/data/test","")
dataset_train.load_data("/data/","train")
dataset_train.prepare()

# Validation dataset
dataset_val = Cells2ChannelDataset()
dataset_val.FIXED_INPUT_SHAPE = True
dataset_val.INPUT_SHAPE = [512, 512]
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
            epochs=650,
            layers='all',
            augmentation=augmentation)
