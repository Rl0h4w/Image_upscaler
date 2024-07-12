import os
import random
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import multiprocessing
from tqdm import tqdm
import asyncio
import loader_videos_YT
from edsr import make_model

def extract_frames_from_video(video_path, frame_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    if not cap.isOpened():
        print(f"Ошибка: Невозможно открыть видеофайл {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Обработка {video_name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frame_dir, f"{video_name}_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        os.remove(video_path)
    else:
        print(f"Ошибка: Не удалось извлечь кадры из {video_path}")

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

def standardize_frames(frames, is_val=False, y_or_x=None):
    frames = frames.astype(np.float32) / 255.0 
    if not is_val:
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(frames.reshape(-1, frames.shape[-1]))
        dump(scaler, f"scaler_model{y_or_x}.joblib")
    else:
        scaler = load(f"scaler_model{y_or_x}.joblib")
        standardized_data = scaler.transform(frames.reshape(-1, frames.shape[-1]))
    return standardized_data.reshape(frames.shape)

def create_dataset(image_dir="data/frames"):
    x_data = []
    y_data = []
    
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    for file_name in files:
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        y_image = cv2.resize(image, (144, 144)) 
        x_image = downscale_and_add_noise(y_image)  
        x_image = cv2.resize(x_image, (36, 36))  
        
        x_data.append(x_image)
        y_data.append(y_image)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
        
    augmentation = data_augmentation()
    
    x_data_aug = augmentation(x_data)
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



if __name__ == "__main__":
    asyncio.run(loader_videos_YT.download())
    extract_frames()
    
    x_data, y_data = create_dataset()
    x_train, x_val, y_train, y_val = split_dataset(x_data, y_data)
    
    x_train = standardize_frames(x_train, is_val=False, y_or_x="x")
    x_val = standardize_frames(x_val, is_val=True, y_or_x="x")
    y_train = standardize_frames(y_train, is_val=False, y_or_x="y")
    y_val = standardize_frames(y_val, is_val=True, y_or_x="y")
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("edsr_model_best.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]
    
    edsr_model = make_model(num_filters=64, num_of_residual_blocks=16)
    optim_edsr = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]
    )
)
    edsr_model.compile(optimizer=optim_edsr, loss="mae", metrics=[psnr_metric, ssim_metric, "accuracy", "mse"])
    edsr_model.fit(x_train, y_train, validation_data = [x_val, y_val], batch_size=2, epochs=100, steps_per_epoch=200, callbacks=callbacks)
