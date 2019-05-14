"""
Mask R-CNN
Configurations and data loading code for cells and cores.

"""

############################################################
#  Configurations
############################################################

import os
import sys
import skimage
from skimage.color import rgb2gray
import numpy as np
import pickle as pkl
import gzip

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class CoresConfig(Config):

    
    MASK_SHAPE = [28, 28]

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 2.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 2.,
        "mrcnn_mask_loss": 2.
    }


    CORE_SUFFIX = "_core" # Suffix for parameters/layer for core model
    MEAN_PIXEL = [ 112.5 ]
    USE_CORE_FEATURES = False  # Train only cores

    NAME = "cores"

    # Train on 3 GPU and 8 original_images per GPU. We can put multiple original_images on each
    # GPU because the original_images are small. Batch size is 8 (GPUs * original_images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + one class

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    USE_MINI_MASK = False

    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 500

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = 'none'

    STEPS_PER_EPOCH = 1000

    DETECTION_MAX_INSTANCES = 200


class CellsConfig(CoresConfig):
    name = "cells"

class Cells2ChannelConfig(CellsConfig):
    MEAN_PIXEL = [ 112.5 , 112.5 ]
    NAME = "input_2channels"
    INPUT_CHANNELS = 2
    # NO_IMAGE_SCALE = True # if use imagenet pretrained model, which has no scaling!

class CellsAndCoresConfig(CoresConfig):
    NAME = "cells_and_cores"
    USE_CORE_FEATURES = True # Use features of core model in training
    USE_BORDER_WEIGHTS = True


class CoresDataset(utils.Dataset):
    """Contains only cores
    """

    FIXED_INPUT_SHAPE = False

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        assert image.ndim == 2 # Grayscale required
        image = np.reshape(image,list(image.shape)+[1])
        return image

    def load_image_vis(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
           Do not convert to one channel array (for visualization)
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        assert image.ndim == 2
        return image

    def load_data(self, dataset_dir, subset):

        self.add_class("core", 1, "core")
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Train or validation dataset?
        assert subset in ["train", "val", "testeval", ""]

        image_dir = os.path.join(dataset_dir,subset,"images")

        for i,img_name in enumerate(next(os.walk(image_dir))[1]):
            image_path = os.path.join(image_dir,img_name,"0.png")
            if not self.FIXED_INPUT_SHAPE:
              image = skimage.io.imread(image_path)
              # assert np.sum(image[:,:,0]) == 0 # First channel must be empty
              height, width = image.shape[:2]
            else:
              height, width = self.INPUT_SHAPE
            self.add_image(
                "core",
                image_id =i,
                image_name=img_name,
                path=image_path,
                height=height,width=width)

    def load_mask(self, image_id):
        """Generate instance gt for an image.
       Returns:
        gt: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance gt.
        """
        info = self.image_info[image_id]
        img_name = info['image_name']
        path_to_masks= self.dataset_dir+"/"+self.subset+"/gt/"+img_name
        mask = []
        for f in next(os.walk(path_to_masks))[2]:
            if f.endswith(".png") and f.startswith("0_"):
                m = skimage.io.imread(os.path.join(path_to_masks, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "core":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class CellsDataset(utils.Dataset):
    """
    Contains only cores
    """

    FIXED_INPUT_SHAPE = False

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        assert image.ndim == 2 # Grayscale required
        image = np.reshape(image,list(image.shape)+[1])
        return image

    def load_image_vis(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
           Do not convert to one channel array (for visualization)
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        assert image.ndim == 2
        return image

    def load_data(self, dataset_dir, subset):

        self.add_class("cell", 1, "cell")
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Train or validation dataset?
        assert subset in ["train", "val", ""]

        image_dir = os.path.join(dataset_dir,subset,"images")

        for i,img_name in enumerate(next(os.walk(image_dir))[1]):
            image_path = os.path.join(image_dir,img_name,"1.png")
            if not self.FIXED_INPUT_SHAPE:
              image = skimage.io.imread(image_path)
              assert np.sum(image[:,:,0]) == 0 # First channel must be empty
              height, width = image.shape[:2]
            else:
              height, width = self.INPUT_SHAPE
            self.add_image(
                "cell",
                image_id =i,
                image_name=img_name,
                path=image_path,
                height=height,width=width)

    def load_mask(self, image_id):
        """Generate instance gt for an image.
       Returns:
        gt: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance gt.
        """
        info = self.image_info[image_id]
        img_name = info['image_name']
        path_to_masks= self.dataset_dir+"/"+self.subset+"/gt/"+img_name
        mask = []
        for f in next(os.walk(path_to_masks))[2]:
            if f.endswith(".png") and f.startswith("1_"):
                m = skimage.io.imread(os.path.join(path_to_masks, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



class CellsWithCoresDataset(CoresDataset):
    """Extends CoreDataset with methods for loading cores and segmentation weights
    """

    def load_core_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['core_path'])
        assert image.ndim == 2
        image = np.reshape(image,list(image.shape)+[1])
        return image

    def load_core_image_vis(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['core_path'])
        assert image.ndim == 2
        return image

    def load_weight_image(self, image_id):
        weights_path = os.path.join(self.dataset_dir, self.subset, "gt",
                                    self.image_info[image_id]["image_name"], "weights","1_weights.pkl")
        with gzip.open(weights_path, 'rb') as f:
            weights = pkl.load(f, encoding='latin1') + 1.
            # add single channel
            weights = np.reshape(weights,list(weights.shape)+[1])
        return weights

    def load_mask(self, image_id):
        """Generate instance gt for an image.
       Returns:
        gt: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance gt.
        """
        # [height, width, instance_count]
        info = self.image_info[image_id]
        img_name = info['image_name']
        path_to_masks= self.dataset_dir+"/"+self.subset+"/gt/"+img_name
        mask = []
        for f in next(os.walk(path_to_masks))[2]:
            if f.endswith(".png") and f.startswith("1_"):
                m = skimage.io.imread(os.path.join(path_to_masks, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_data(self, dataset_dir, subset):

        self.add_class("cell", 1, "cell")
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Train or validation dataset?
        assert subset in ["train", "val", ""]

        image_dir = os.path.join(dataset_dir,subset,"images")

        for i,img_name in enumerate(next(os.walk(image_dir))[1]):
            image_path = os.path.join(image_dir,img_name,"1.png")
            core_path = os.path.join(image_dir,img_name,"0.png")
            if not self.FIXED_INPUT_SHAPE:
              image = skimage.io.imread(image_path)
              # assert np.sum(image[:,:,0]) == 0 # First channel must be empty
              height, width = image.shape[:2]
            else:
              height, width = self.INPUT_SHAPE
            self.add_image(
                "cell",
                image_id=i,
                image_name=img_name,
                path=image_path,
                core_path=core_path,
                height=height,width=width)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




class CoresDSB18Dataset(utils.Dataset):
    """Contains only cores
    """
    def load_core_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['core_path'])
        # assert image.ndim == 2
        image = rgb2gray(image)
        image = np.reshape(image,list(image.shape)+[1])
        return image

    def load_core_image_vis(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['core_path'])
        #assert image.ndim == 2
        return image


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # assert image.ndim == 2 # Grayscale required
        image = rgb2gray(image)
        image = np.reshape(image,list(image.shape)+[1])
        return image

    def load_image_vis(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
           Do not convert to one channel array (for visualization)
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        #assert image.ndim == 2
        return image

    def load_data(self, dataset_dir, subset):

        self.add_class("core", 1, "core")
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Train or validation dataset?
        assert subset in ["train", "val", "testeval", ""]

        image_dir = os.path.join(dataset_dir,subset)

        for i,img_name in enumerate(next(os.walk(image_dir))[1]):
            image_path = os.path.join(image_dir,img_name,"images",img_name+".png")
            core_path = image_path
            if not self.FIXED_INPUT_SHAPE:
              image = skimage.io.imread(image_path)
              assert np.sum(image[:,:,0]) == 0 # First channel must be empty
              height, width = image.shape[:2]
            else:
              height, width = self.INPUT_SHAPE
            self.add_image(
                "core",
                image_id =i,
                image_name=img_name,
                path=image_path,
                core_path=core_path,
                height=height,width=width)

    def load_mask(self, image_id):
        """Generate instance gt for an image.
       Returns:
        gt: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance gt.
        """
        info = self.image_info[image_id]
        img_name = info['image_name']
        path_to_masks= self.dataset_dir+"/"+self.subset+"/"+img_name+"/masks"
        mask = []
        for f in next(os.walk(path_to_masks))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(path_to_masks, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "core":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



class Cells2ChannelDataset(utils.Dataset):

    FIXED_INPUT_SHAPE = False
    INPUT_SHAPE = [ 512, 512]
    # img[:,:,1] = img_cell[:,:]
    # img[:,:,2] = img_core[:,:]
    # img[:,:,0] = 0

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,1] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        assert image.ndim == 3 # rgb required
        #image = np.reshape(image,list(image.shape)+[1])
        image = image[:,:,[1,2]] # first channel should be black
        assert image.shape[2] == 2
        return image

    def load_image_vis(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
           Do not convert to one channel array (for visualization)
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        #assert image.ndim == 2
        return image

    def load_data(self, dataset_dir, subset):

        self.add_class("cell", 1, "cell")
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Train or validation dataset?
        assert subset in ["train", "val", ""]

        image_dir = os.path.join(dataset_dir,subset,"images")

        for i,img_name in enumerate(next(os.walk(image_dir))[1]):
            image_path = os.path.join(image_dir,img_name,"rgb.png")
            if not self.FIXED_INPUT_SHAPE:
              image = skimage.io.imread(image_path)
              assert np.sum(image[:,:,0]) == 0 # First channel must be empty
              height, width = image.shape[:2]
            else:
              height, width = self.INPUT_SHAPE
            self.add_image(
                "cell",
                image_id =i,
                image_name=img_name,
                path=image_path,
                height=height,width=width)

    def load_mask(self, image_id):
        """Generate instance gt for an image.
       Returns:
        gt: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance gt.
        """
        info = self.image_info[image_id]
        img_name = info['image_name']
        path_to_masks= self.dataset_dir+"/"+self.subset+"/gt/"+img_name
        mask = []
        for f in next(os.walk(path_to_masks))[2]:
            if f.endswith(".png") and f.startswith("1_"):
                m = skimage.io.imread(os.path.join(path_to_masks, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
