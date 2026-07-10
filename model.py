"""
model.py
--------
Defines a lightweight, mini-Xception-style CNN for facial emotion
classification. Designed to be small and fast enough for real-time
inference on CPU, while still using depthwise-separable residual blocks
for reasonable accuracy on FER2013-style 48x48 grayscale images.
"""

from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    SeparableConv2D, GlobalAveragePooling2D, Dropout, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def _residual_separable_block(x, filters, l2_reg):
    """One residual block: two separable convs + a projected skip connection."""
    residual = Conv2D(filters, (1, 1), strides=(2, 2), padding="same",
                       use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(filters, (3, 3), padding="same",
                         depthwise_regularizer=l2(l2_reg),
                         pointwise_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(filters, (3, 3), padding="same",
                         depthwise_regularizer=l2(l2_reg),
                         pointwise_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = Add()([x, residual])
    x = Dropout(0.3)(x)
    return x


def build_emotion_model(input_shape=(48, 48, 1), num_classes=7, l2_reg=1e-2):
    """
    Builds and returns the (uncompiled) Keras model.

    Args:
        input_shape: shape of the input grayscale face crop.
        num_classes: number of emotion categories to predict.
        l2_reg: L2 weight regularization strength.

    Returns:
        tf.keras.Model
    """
    img_input = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(l2_reg),
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(l2_reg),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for filters in (16, 32, 64, 128):
        x = _residual_separable_block(x, filters, l2_reg)

    x = Conv2D(num_classes, (3, 3), padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation("softmax", name="predictions")(x)

    model = Model(inputs=img_input, outputs=output, name="mini_emotion_cnn")
    return model


EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]


if __name__ == "__main__":
    # Quick sanity check: build the model and print a summary.
    m = build_emotion_model()
    m.summary()
    print(f"Total parameters: {m.count_params():,}")
