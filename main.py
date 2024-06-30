import os
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import multiprocessing

def extract_frames_from_video(video_path, frame_dir):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"{video_name}_{frame_count:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    os.remove(video_path)

def extract_frames(video_dir="data/videos", frame_dir="data/frames"):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    
    with multiprocessing.Pool() as pool:
        pool.starmap(extract_frames_from_video, [(video_path, frame_dir) for video_path in files])

def downscale_and_add_noise(image):
    brightness_factor = random.uniform(0.5, 1.5)
    bright_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    hsv_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2HSV)
    saturation_factor = random.uniform(0.5, 1.5)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    sat_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    contrast_factor = random.uniform(0.5, 1.5)
    contrast_image = cv2.convertScaleAbs(sat_image, alpha=contrast_factor, beta=0)
    
    noise_factor = random.uniform(0, 0.05)
    noise = np.random.randn(*contrast_image.shape) * 255 * noise_factor
    noisy_image = cv2.add(contrast_image, noise.astype(np.uint8))
    
    scale_factor = random.uniform(0.2, 1.0)
    height, width = noisy_image.shape[:2]
    low_res_image = cv2.resize(noisy_image, (int(width * scale_factor), int(height * scale_factor)))
    
    return low_res_image

def data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

def standardize_frames(frames):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(frames.reshape(-1, frames.shape[-1]))
    dump(scaler, "scaler_model.joblib")
    return standardized_data.reshape(frames.shape)

def create_dataset(image_dir="data/images"):
    x_data = []
    y_data = []
    
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    for file_name in files:
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        y_image = cv2.resize(image, (720, 720)) 
        x_image = downscale_and_add_noise(y_image)  
        x_image = cv2.resize(x_image, (360, 360))  
        
        x_data.append(x_image)
        y_data.append(y_image)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    x_data_standardized = standardize_frames(x_data)
    
    augmentation = data_augmentation()
    
    x_data_aug = augmentation(x_data_standardized)
    y_data_aug = augmentation(y_data)
    return x_data_aug, y_data_aug

def split_dataset(x_data, y_data, test_size=0.2):
    split_idx = int(len(x_data) * (1 - test_size))
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    return x_train, x_val, y_train, y_val

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(c9)
    
    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy", psnr_metric, ssim_metric])
    return model

if __name__ == "__main__":
    extract_frames()
    x_data, y_data = create_dataset()
    x_train, x_val, y_train, y_val = split_dataset(x_data, y_data)
    
    unet = unet_model((360, 360, 3))
    unet.summary()
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("unet_model_best.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]
    
    unet.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_val, y_val), callbacks=callbacks)
    