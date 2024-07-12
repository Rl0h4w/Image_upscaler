import tensorflow as tf
import keras
from keras import layers
from keras import ops

dataset_name = "oxford_flowers102"
dataset_repetitions = 5
num_epochs = 1  
image_size = 64

kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

min_signal_rate = 0.02
max_signal_rate = 0.95

embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


@keras.saving.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        self.encoder = keras.Sequential(
            [keras.Input(shape=(image_size, image_size, 3)),
             layers.Rescaling(255.),
             layers.Resizing(height=kid_image_size, width=kid_image_size),
             layers.Lambda(keras.applications.inception_v3.preprocess_input),
             keras.applications.InceptionV3(
                 include_top=False,
                 input_shape=(kid_image_size, kid_image_size, 3),
                 weights="imagenet",
                 ),
             layers.GlobalAveragePooling2D(),
             ],
            name="inception_encoder",
        )
    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], dtype="float32")
        return (features_1 @ ops.transpose(features_2)/feature_dimensions + 1.0) **3.0