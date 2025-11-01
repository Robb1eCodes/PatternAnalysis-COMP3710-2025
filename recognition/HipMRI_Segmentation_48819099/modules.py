import tensorflow as tf

def double_conv(tensor, filters, conv_size = 3, activation = 'relu'):
    """
    Applies two convolutional layers, each followed by batch normalization and activation.

    Args:
        tensor: Input tensor to the block (feature set from previous layer).
        filters: Number of filters in the Conv2D layers.
        conv_size: Size of the convolution (default to 3).
        activation: Activation function to use (default to 'relu').

    Returns:
        Output tensor after two convolutions, batch norm, and activation.
    """

    tensor = tf.keras.layers.Conv2D(filters, conv_size, padding='same')(tensor)
    tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.layers.Activation(activation)(tensor)
    tensor = tf.keras.layers.Conv2D(filters, conv_size, padding='same')(tensor)
    tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.layers.Activation(activation)(tensor)

    return tensor


def unet_2d(input_size = (256, 256, 1), num_filters=32):
    """
    Builds a 2D U-Net model to mask Hip MRI study data.

    Args:
        input_size: Shape of input images (height, width, channels).
        num_filters: Number of filters in the first layer.

    Returns:
        A compiled Keras Model representing the U-Net architecture.
    """

    # Input layer: expects images of shape input_size
    inputs = tf.keras.Input(shape=input_size)

    # Encoder (downsampling)
    # Block 1
    c1 = double_conv(inputs, num_filters)  # Feature extraction
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)  # Downsample (Using MaxPooling)

    # Block 2
    c2 = double_conv(p1, num_filters * 2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Block 3
    c3 = double_conv(p2, num_filters * 4)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Block 4
    c4 = double_conv(p3, num_filters * 8)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck (End of Encoder)
    c5 = double_conv(p4, num_filters * 16)

    # Decoder (upsampling)
    # Up Block 4
    u4 = tf.keras.layers.Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)  # Upsample
    u4 = tf.keras.layers.concatenate([u4, c4])  # Skip connection
    c6 = double_conv(u4, num_filters * 8)

    # Up Block 3
    u3 = tf.keras.layers.Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c7 = double_conv(u3, num_filters * 4)

    # Up Block 2
    u2 = tf.keras.layers.Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c8 = double_conv(u2, num_filters * 2)

    # Up Block 1
    u1 = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c9 = double_conv(u1, num_filters)

    # Output layer: produces segmentation mask (6 channels (classes), softmax activation)
    outputs = tf.keras.layers.Conv2D(6, (1, 1), activation='softmax')(c9)

    # Build model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == '__main__':
    """
    Creates the model object, does not run any data through. 

    Provides a summary of the model to check everything has been setup correctly.
    """

    m = unet_2d((128, 128, 1))
    m.summary()