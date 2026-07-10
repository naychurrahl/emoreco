"""
train.py
--------
Trains the emotion recognition CNN on FER2013 (Kaggle folder-structured
version: a 'train' folder and a 'test' folder, each containing one
subfolder of images per emotion).

Usage:
    python train.py --data_dir fer2013 --epochs 50 --batch_size 64
    (where fer2013/ contains 'train' and 'test' subfolders)
"""

import argparse
import json

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam

from model import build_emotion_model
from preprocess import load_fer2013_from_folders, compute_class_weights_from_directory


def parse_args():
    p = argparse.ArgumentParser(description="Train facial emotion recognition CNN")
    p.add_argument("--data_dir", type=str, default="fer2013",
                    help="Path to folder containing 'train' and 'test' subfolders")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=str, default="best_model.keras",
                    help="Where to save the best checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading FER2013 from '{args.data_dir}' ...")
    train_ds, val_ds, test_ds, class_names = load_fer2013_from_folders(
        args.data_dir, batch_size=args.batch_size
    )
    print("Detected classes (in model output order):", class_names)

    # Save the class order so realtime_emotion.py always matches this run's
    # model, regardless of how the folders happened to sort.
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    class_weights = compute_class_weights_from_directory(args.data_dir, class_names)
    print("Class weights (to counter imbalance):", class_weights)

    model = build_emotion_model(input_shape=(48, 48, 1),
                                 num_classes=len(class_names))
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(args.output, monitor="val_accuracy",
                         save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                           patience=4, min_lr=1e-6, verbose=1),
        CSVLogger("training_log.csv"),
    ]

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")
    print(f"Best model saved to: {args.output}")
    print("Class name order saved to: class_names.json")


if __name__ == "__main__":
    main()
