import os
import random
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import multiprocessing
from tqdm import tqdm
import loader_videos_YT
from models import get_combined_model

x_shape = 32
scale = 4
y_shape = x_shape * scale

def extract_frames_from_video(video_path, frame_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_paths = []

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frame_dir, f"{video_name}_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])  
            frame_paths.append(frame_path)
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_paths:
        # os.remove(video_path)
        pass
    else:
        print(f"Error: Unable to extract frames from {video_path}")

def extract_frames(video_dir="data/videos", frame_dir="data/frames"):
    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} does not exist.")
        return
    
    os.makedirs(frame_dir, exist_ok=True)
    
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    
    with multiprocessing.Pool() as pool:
        pool.starmap(extract_frames_from_video, [(video_path, frame_dir) for video_path in files])

def downscale_and_add_noise(image):
    brightness_factor = random.uniform(0.5, 1.5)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_factor = random.uniform(0.5, 1.5)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    contrast_factor = random.uniform(0.5, 1.5)
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    
    noise_factor = random.uniform(0, 0.05)
    noise = np.random.randn(*image.shape) * 255 * noise_factor
    image = cv2.add(image, noise.astype(np.uint8))
    
    if random.random() < 0.5:  
        ksize = random.choice([(3, 3), (5, 5), (7, 7)])
        image = cv2.GaussianBlur(image, ksize, 0)
    
    if random.random() < 0.5:  
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(10, 50)]
        _, encoded_image = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(encoded_image, 1)
    
    scale_factor = random.uniform(0.2, 1.0)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    
    image = cv2.resize(image, (width, height))
    
    return image

def data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

def standardize_frame(frame, scaler):
    frame = frame.astype(np.float32) / 255.0
    return scaler.transform(frame.reshape(-1, frame.shape[-1])).reshape(frame.shape)

def frame_generator(files, batch_size, scaler, is_val=False):
    while True:
        random.shuffle(files)
        for start in range(0, len(files), batch_size):
            x_data = []
            y_data = []
            end = min(start + batch_size, len(files))
            for file_name in files[start:end]:
                image_path = file_name
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                y_image = cv2.resize(image, (y_shape, y_shape))
                x_image = downscale_and_add_noise(y_image.copy())
                x_image = cv2.resize(x_image, (x_shape, x_shape))
                
                x_image = standardize_frame(x_image, scaler)
                y_image = standardize_frame(y_image, scaler)
                
                x_data.append(x_image)
                y_data.append(y_image)
            
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            yield x_data, y_data

def split_file_paths(files, test_size=0.2):
    split_idx = int(len(files) * (1 - test_size))
    return files[:split_idx], files[split_idx:]

def prepare_scaler(image_dir):
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    sample_images = []
    for file_name in files[:1000]: 
        image_path = file_name
        image = cv2.imread(image_path)
        if image is None:
            continue
        y_image = cv2.resize(image, (y_shape, y_shape))
        x_image = downscale_and_add_noise(y_image.copy())
        x_image = cv2.resize(x_image, (x_shape, x_shape))
        sample_images.append(x_image)
        sample_images.append(y_image)
    sample_images = [cv2.resize(img, (x_shape, x_shape)) if img.shape[:2] == (y_shape, y_shape) else img for img in sample_images]
    sample_images = np.array(sample_images).astype(np.float32) / 255.0
    scaler = StandardScaler().fit(sample_images.reshape(-1, sample_images.shape[-1]))
    return scaler

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

if __name__ == "__main__":
    if not os.path.exists("data/videos") and not os.path.exists("data/frames"):
        loader_videos_YT.download()
        extract_frames()
    
    batch_size = 32
    image_dir = "data/frames"
    
    scaler = prepare_scaler(image_dir)
    
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    train_files, val_files = split_file_paths(files, test_size=0.2)
    
    train_gen = frame_generator(train_files, batch_size, scaler, is_val=False)
    val_gen = frame_generator(val_files, batch_size, scaler, is_val=True)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("unet_model_best.keras", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb]
    
    model = get_combined_model(scale=scale, input_shape=(x_shape, x_shape, 3))
    
    model.compile(optimizer='adam', loss='mse', metrics=["mae", "mse", "accuracy", psnr_metric, ssim_metric])
    
    model.fit(
        train_gen,
        steps_per_epoch=len(train_files) // batch_size,
        validation_data=val_gen,
        validation_steps=len(val_files) // batch_size,
        epochs=50,
        callbacks=callbacks
    )