"""
preprocess.py
-------------
Loads FER2013 when provided as Kaggle's folder-structured version:

    fer2013/
      train/
        angry/      *.jpg
        disgust/    *.jpg
        fear/       *.jpg
        happy/      *.jpg
        neutral/    *.jpg
        sad/        *.jpg
        surprise/   *.jpg
      test/
        angry/ ...
        ... (same subfolders)

Builds tf.data.Dataset pipelines (train/val/test), with the train folder
further split into train/validation. Also computes class weights from
folder image counts to counter FER2013's class imbalance (Disgust has far
fewer images than the others).
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64


def load_fer2013_from_folders(data_dir, batch_size=BATCH_SIZE,
                               val_split=0.1, seed=123):
    """
    Args:
        data_dir: path to the folder containing 'train' and 'test' subfolders.
        batch_size: batch size for all three datasets.
        val_split: fraction of the train folder held out for validation.
        seed: shuffle seed, kept fixed so train/val split is reproducible.

    Returns:
        train_ds, val_ds, test_ds: batched, normalized tf.data.Dataset objects
        class_names: list of class names in the index order used by the
                     model's output (alphabetical, as assigned by Keras).
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", seed=seed,
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", seed=seed,
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, label_mode="categorical",
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, label_mode="categorical", shuffle=False,
    )

    class_names = train_ds.class_names  # e.g. ['angry','disgust','fear',...]

    normalization = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    augmenter = build_augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights_from_directory(data_dir, class_names):
    """Weights inversely proportional to class frequency, based on how many
    image files sit in each train/<class_name>/ folder."""
    train_dir = os.path.join(data_dir, "train")
    counts = np.array(
        [len(os.listdir(os.path.join(train_dir, name))) for name in class_names],
        dtype=np.float32,
    )
    total = counts.sum()
    n_classes = len(counts)
    weights = total / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_augmentation_layer():
    """Augmentation applied only to the training split (rotation ~10deg,
    shift/zoom ~10%, horizontal flip)."""
    return keras.Sequential([
        keras.layers.RandomRotation(0.028),   # ~10 degrees (fraction of 2*pi)
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomFlip("horizontal"),
    ], name="augmentation")
