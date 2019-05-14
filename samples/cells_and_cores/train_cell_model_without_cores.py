import os
import imgaug
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join("/data/logs")


# Directory of images to run detection on
IMAGE_DIR = os.path.join("/data/")

from samples.cells_and_cores.cells_and_cores import CellsConfig

config = CellsConfig()
config.BATCH_SIZE = 2
config.IMAGE_MIN_DIM = 512
config.IMAGE_MAX_DIM = 512
config.IMAGES_PER_GPU = 2
config.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) # (8, 16, 32, 64, 128)
config.USE_BORDER_WEIGHTS = False #True
config.USE_CORE_FEATURES = False #True
config.LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 2.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 2.,
        "mrcnn_mask_loss": 2.
    }


config.display()

# Create core model object
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR+"/cells_with_cores", config=config)



layernames = [ l.name for l in model.keras_model.layers ]
#print (len(layernames))
#print (layernames)
model.load_weights("/data/models/model.e120.hdf5", by_name=True,exclude=['avg_pool','fc1000'])

from samples.cells_and_cores.cells_and_cores import CellsWithCoresDataset

# Training dataset
dataset_train = CellsWithCoresDataset()
#dataset_train.load_data("/data/test","")
dataset_train.load_data("/data/","train")
dataset_train.prepare()

# Validation dataset
dataset_val = CellsWithCoresDataset()
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
            learning_rate=config.LEARNING_RATE / 10,
            epochs=650,
            layers='all',
            augmentation=augmentation)
