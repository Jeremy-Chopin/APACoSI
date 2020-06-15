import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
import math 
from Helper_files.Dataloader import *
from Helper_files.Dataset import *


def load_model(model_path, classes):

    BACKBONE = 'efficientnetb3'
    LR = 0.0001

    # define network parameters
    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 352, 3))
    
    # load best weights
    model.load_weights(model_path) 

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # defin loss
    dice_loss = sm.losses.DiceLoss() 
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # define metrics
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    return model

def load_dataset(directory_rgb, directory_labels, classes):

    BACKBONE = 'efficientnetb3'

    preprocess_input = sm.get_preprocessing(BACKBONE)

    test_dataset = Dataset(
        directory_rgb, 
        directory_labels, 
        classes=classes, 
        augmentation=None,
        preprocessing=get_preprocessing(preprocess_input)
    )

    return test_dataset

def get_segmentation(image, model):

    image_to_predict = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image_to_predict)[0]

    return pr_mask

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=352, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=352, width=512, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(352, 512)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def predict_image_segmentation(directory_rgb, directory_labels, classes):

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    BACKBONE = 'efficientnetb3'
    LR = 0.0001

    # define network parameters
    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 352, 3))
    preprocess_input = sm.get_preprocessing(BACKBONE)

    test_dataset = Dataset(
        directory_rgb, 
        directory_labels, 
        classes=classes, 
        augmentation=None,
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights('best_model.h5') 

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss() 
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    image, gt_mask = test_dataset[5]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    return (image[0], pr_mask[0])

def reduce_pmap_depth(pmap):

    x,y,z = pmap.shape

    temp = np.zeros((x,y))

    for i in range(0,x):
        for j in range(0,y):
            temp[i][j] = __reduce_depth_metrics(pmap[i][j])
    
    return temp

def __reduce_depth_metrics(depth_vector):

    index = 0
    value = 0

    for i in range(0, len(depth_vector)):
        if value < depth_vector[i]:
            index = i
            value = depth_vector[i]
    
    return index

def reduce_pmap_depth_confusion(pmap, value_conf, seuil):

    x,y,z = pmap.shape

    temp = np.zeros((x,y))

    for i in range(0,x):
        for j in range(0,y):
            temp[i][j] = __reduce_depth_metrics_confusion(pmap[i][j], value_conf, seuil)
    
    return temp

def __reduce_depth_metrics_confusion(depth_vector, value_conf, seuil):

    index = 0
    value = 0
    precedent_index = 0

    for i in range(0, len(depth_vector)):
        if value < depth_vector[i]:
            precedent_index = index
            index = i
            value = depth_vector[i]
    
    step = math.fabs(depth_vector[i] - depth_vector[index])

    if  step < seuil:
        return value_conf
    else:
        return index