import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders

DATA_DIR = "data/raw"

SPLIT_DATA_DIR = "data/data_split"

splitfolders.ratio(DATA_DIR, output = SPLIT_DATA_DIR, seed=42, ratio=(.7, .15, .15))

train_path = "data/data_split/train"
test_path = "data/data_split/test"
valid_path = "data/data_split/val"


train_data_gen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,           
    width_shift_range=0.2,       
    height_shift_range=0.2,      
    shear_range=0.2,             
    zoom_range=0.2,             
    horizontal_flip=True,        
    fill_mode='nearest')

test_data_gen = ImageDataGenerator(rescale=1./255)
valid_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

valid_generator = valid_data_gen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_generator = test_data_gen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')