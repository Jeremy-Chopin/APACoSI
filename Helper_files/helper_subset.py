import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
from Helper_files.helper import *
from Helper_files.Dataloader import *
from Helper_files.Dataset import *


def train_model(DATA_DIR, CLASSES, model_path, GPU):
    
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x_train_dir = os.path.join(DATA_DIR, 'Train_RGB')
    y_train_dir = os.path.join(DATA_DIR, 'Train_Labels')

    x_valid_dir = os.path.join(DATA_DIR, 'Val_RGB')
    y_valid_dir = os.path.join(DATA_DIR, 'Val_Labels')

    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 2
    CLASSES = CLASSES
    LR = 0.0001
    EPOCHS = 50

    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 352, 3))

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

    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        classes=CLASSES, 
        augmentation=None,
        preprocessing=sm.get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES, 
        augmentation=None,
        preprocessing=sm.get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
        keras.callbacks.EarlyStopping(monitor="val_loss")
    ]

    # print model
    print(model.summary())

    # train model
    history = model.fit_generator(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader),
    )