import tensorflow as tf
from keras import layers, Model

class DepthToSpaceLayer(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(DepthToSpaceLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

def edsr(scale, num_filters=128, num_res_blocks=32, res_block_scaling=None):
    def res_block(x_in, num_filters, scaling):
        x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
        if scaling:
            x = layers.Lambda(lambda t: t * scaling)(x)
        x = layers.Add()([x_in, x])
        return x

    x_in = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x_in)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters, res_block_scaling)

    x = layers.Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = layers.Conv2D(3, kernel_size=3, padding='same')(x)
    x = layers.Add()([x_in, x])

    x = layers.Conv2D(3 * (scale ** 2), kernel_size=3, padding='same')(x)
    x = DepthToSpaceLayer(scale)(x)

    model = Model(x_in, x)
    return model

def unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs, outputs)
    return model

def get_combined_model(scale, input_shape):
    edsr_model = edsr(scale=scale)
    unet_model = unet(input_shape=(input_shape[0] * scale, input_shape[1] * scale, 3))

    inputs = layers.Input(shape=input_shape)
    edsr_output = edsr_model(inputs)
    outputs = unet_model(edsr_output)

    model = Model(inputs, outputs)
    return model

