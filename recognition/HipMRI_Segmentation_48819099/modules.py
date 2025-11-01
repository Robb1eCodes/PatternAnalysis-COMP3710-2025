import tensorflow as tf

def conv_block(tensor, filters, conv_size = 3, activation = 'relu'):
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
    tensor = tf.keras.Activation(activation)(tensor)
    tensor = tf.keras.layers.Conv2D(filters, conv_size, padding='same')(tensor)
    tensor = tf.keras.layers.BatchNormalization()(tensor)
    tensor = tf.keras.Activation(activation)(tensor)

    return tensor
