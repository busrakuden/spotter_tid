import os
import cv2
import imageio
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

def remove_background_from_videos(
    video_file, 
    save_path,
    crop_sides=0.2,
    channel=cv2.COLOR_BGR2RGB,
):
    video_id = video_file.split('/')[-1].replace('.mp4', '')
    
    if os.path.exists(os.path.join(save_path, f"{video_id}.gif")):
        print(f"Background removed GIF for {video_id} already exists. Skipping...")
        return
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    cap = cv2.VideoCapture(video_file)
    
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    frames = []
    while True:

        ret, img = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if h != w:
            # Center crop to make it square
            min_dim = min(h, w)
            start_x = (w - min_dim + 1) // 2
            start_y = (h - min_dim + 1) // 2

            rgb = rgb[start_y:start_y+min_dim, start_x:start_x+min_dim]
            img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

        result = segmentor.process(rgb)
        mask = result.segmentation_mask > 0.7
        
        bg_removed = img.copy()
        # gray mask
        bg_removed[~mask] = [128, 128, 128]  # or [0,0,0] for black

        # Resize to 256x256
        bg_removed = cv2.resize(bg_removed, (256, 256))
        
        bg_removed = cv2.cvtColor(bg_removed, channel)

        frames.append(bg_removed)

    imageio.mimwrite(os.path.join(save_path, f"{video_id}.gif"), frames, fps=25, loop=0)

    cap.release()
    
VIDEO_PATH = {
    'tid_sozluk': '/home/ks0085/Documents/Datasets/tid/videos/tid_sozluk',
    'isaretce': '/home/ks0085/Documents/Datasets/tid/videos/isaretce',
    'afid': '/home/ks0085/Documents/Datasets/tid/videos/afid',
}

for dataset, path in VIDEO_PATH.items():
    save_path = f'/home/ks0085/Documents/notebooks/butid/dictionary/rgb/{dataset}'
    video_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')]
    
    crop_sides = 0.01
    for video_file in tqdm(video_files, desc=f"Processing {dataset}"):
        if dataset == 'tid_sozluk':
            crop_sides = 0.2
            channel = cv2.COLOR_BGR2RGBA
        else:
            channel = cv2.COLOR_BGR2RGB

        try:
            remove_background_from_videos(video_file, save_path, crop_sides=crop_sides, channel=channel)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")