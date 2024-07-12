import tensorflow as tf
import numpy as np
import keras
from keras import layers
class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(tf.cast(super_resolution_img, tf.unit8), axis=0)
        return super_resolution_img

def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])

def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x

def make_model(num_filters, num_of_residual_blocks):
    input_layer = layers.Input(None, None, 3)
    x = layers.Rescaling(scale=1.0/255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)
    
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)
    
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    
    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)
    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)