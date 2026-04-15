import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import math

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("Hata: MoviePy yüklenemedi.")

from src.utils import load_model, load_rgb_video, prepare_input, sliding_windows

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
CROPPED_VIDEOS_DIR = Path(r"C:\Users\busra\Sign_Language_Project\organized_crops\tid_sozluk")

# --- GÜNCELLEME: Çıktı yolları "3. Video" için ayarlandı ---
OUTPUT_MP4_DIR = Path("data/butid/dictionary/rgb/new_additions8_i3d_mp4")
OUTPUT_CSV = "spotting_results_8rd_videos_mp4.csv"

CHECKPOINT_PATH = Path("models/i3d/i3d_mlp.pth.tar") 
DICTIONARY_EMB_PATH = Path("data/butid/dictionary/i3d/tid_sozluk8.i3d.pth")
DICTIONARY_CSV_PATH = Path("data/dictionary.csv") 

# --- HIZLANDIRMA AYARLARI ---
ARCH = "i3d_mlp"
BATCH_SIZE = 40       
WINDOW_STRIDE = 4     
NUM_IN_FRAMES = 16 
FPS = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# ==============================================================================
# 2. YARDIMCI FONKSİYONLAR
# ==============================================================================

def load_gloss_to_id_map(csv_path):
    df = pd.read_csv(csv_path, dtype={'video_id': str})
    gloss_map = {}
    for _, row in df.iterrows():
        gloss = row['gloss']
        vid_id = str(row['video_id']).strip()
        if '.' in vid_id: root_id = vid_id.split('.')[0]
        else: root_id = vid_id
        gloss_map[gloss] = root_id.zfill(4) 
    print(f"📄 CSV Sözlük Yüklendi: {len(gloss_map)} kelime ID'si haritalandı.")
    return gloss_map

def get_next_filename(output_dir, root_id):
    i = 0
    while True:
        filename = f"{root_id}.{str(i).zfill(3)}.mp4"
        path = output_dir / filename
        if not path.exists(): return path
        i += 1

def save_clip_as_mp4(video_path, output_path, center_time, duration_sec=1.0):
    try:
        clip = VideoFileClip(str(video_path))
        start_t = max(0, center_time - (duration_sec / 2))
        end_t = min(clip.duration, center_time + (duration_sec / 2))
        
        if end_t - start_t < 0.2: 
            clip.close()
            return False

        if hasattr(clip, 'subclipped'): new_clip = clip.subclipped(start_t, end_t)
        else: new_clip = clip.subclip(start_t, end_t)
            
        if hasattr(new_clip, 'resized'): new_clip = new_clip.resized(height=224)
        else: new_clip = new_clip.resize(height=224)
        
        new_clip.write_videofile(str(output_path), codec='libx264', audio=False, fps=FPS, logger=None)
        clip.close()
        return True
    except Exception as e:
        print(f"MP4 Kayıt Hatası ({video_path.name}): {e}")
        return False

def load_dictionary(dict_path):
    print(f"📚 Embedding Sözlüğü yükleniyor: {dict_path}")
    data = torch.load(dict_path, map_location='cpu', weights_only=False)
    glosses = data['glosses']
    embeddings = data['embeddings']
    
    if isinstance(embeddings, list): embeddings = np.array(embeddings)
    if isinstance(glosses, list): glosses = np.array(glosses)
    
    unique_glosses = np.unique(glosses)
    prototype_dict = {}
    for gloss in unique_glosses:
        indices = np.where(glosses == gloss)[0]
        feats = embeddings[indices]
        mean_feat = np.mean(feats, axis=0)
        mean_feat = mean_feat / np.linalg.norm(mean_feat)
        prototype_dict[gloss] = mean_feat
    print(f"   -> {len(prototype_dict)} adet tekil gloss prototipi oluşturuldu.")
    return prototype_dict

def get_spotting_prediction(model, video_path, target_embedding):
    try:
        rgb = load_rgb_video(video_path, fps=FPS)
        rgb = prepare_input(rgb)
        rgb_slides, t_mid = sliding_windows(rgb=rgb, stride=WINDOW_STRIDE, num_in_frames=NUM_IN_FRAMES)
        
        num_clips = rgb_slides.shape[0]
        if num_clips == 0: return None
        
        num_batches = math.ceil(num_clips / BATCH_SIZE)
        continuous_features = []
        with torch.no_grad():
            for b in range(num_batches):
                inp = rgb_slides[b * BATCH_SIZE : (b + 1) * BATCH_SIZE].to(DEVICE)
                out = model(inp)
                continuous_features.append(out["embds"].cpu().numpy())
        
        continuous_features = np.vstack(continuous_features)
        target = target_embedding.reshape(1, -1)
        dists = pairwise_distances(continuous_features, target, metric="cosine")
        sims = 1 - dists / 2
        
        max_sim = np.max(sims)
        peak_idx = np.argmax(sims)
        peak_frame = t_mid[peak_idx]
        peak_time_sec = peak_frame / FPS
        
        return {"max_score": max_sim, "peak_time": peak_time_sec, "frame_idx": peak_frame}
    except Exception as e:
        print(f"Hata ({video_path.name}): {e}")
        return None

# ==============================================================================
# 3. ANA DÖNGÜ
# ==============================================================================

def main():
    if not OUTPUT_MP4_DIR.exists():
        os.makedirs(OUTPUT_MP4_DIR)

    prototypes = load_dictionary(DICTIONARY_EMB_PATH) 
    gloss_id_map = load_gloss_to_id_map(DICTIONARY_CSV_PATH) 
    
    print(f"🧠 Model Yükleniyor: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH, arch=ARCH) 
    model = model.to(DEVICE)
    model.eval()
    
    if not CROPPED_VIDEOS_DIR.exists():
        print(f"HATA: Giriş klasörü bulunamadı -> {CROPPED_VIDEOS_DIR}")
        return

    gloss_folders = [f for f in CROPPED_VIDEOS_DIR.iterdir() if f.is_dir()]
    gloss_folders.sort(key=lambda f: f.name)

    print(f"🚀 Üçüncü video için spotting işlemi başlıyor (ID <= 500, Çıktı: MP4)...")
    
    for folder in tqdm(gloss_folders, desc="İşleniyor"):
        gloss_name = folder.name
        
        if gloss_name not in prototypes: continue
        if gloss_name not in gloss_id_map: continue

        root_id = gloss_id_map[gloss_name] 
        
        # ID <= 500 Filtresi
        try:
            if int(root_id) > 300:
                continue
        except ValueError:
            continue

        target_emb = prototypes[gloss_name]
        
        video_files = list(folder.glob("*.mp4"))
        
        # --- GÜNCELLEME: En az 3 video kontrolü ---
        if len(video_files) < 8:
            continue
        
        video_files.sort(key=lambda f: f.name.lower()) 
        
        # --- GÜNCELLEME: 3. Videoyu al (Index 2) ---
        vid_path = video_files[7] 
        
        prediction = get_spotting_prediction(model, vid_path, target_emb)
        
        if prediction:
            score = prediction["max_score"]
            timestamp = prediction["peak_time"]
            
            save_path = get_next_filename(OUTPUT_MP4_DIR, root_id)
            
            success = save_clip_as_mp4(vid_path, save_path, timestamp, duration_sec=1.0)
            
            if success:
                new_row = {
                    "gloss": gloss_name,
                    "id": root_id,
                    "original_video": vid_path.name,
                    "saved_as": save_path.name,
                    "score": score,
                    "timestamp": timestamp,
                    "note": "5rd_video_mp4" # Not güncellendi
                }
                
                df_row = pd.DataFrame([new_row])
                file_exists = os.path.exists(OUTPUT_CSV)
                df_row.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)

    print(f"\n✅ İşlem bitti! Sonuçlar '{OUTPUT_CSV}' dosyasına anlık olarak eklendi.")

if __name__ == "__main__":
    main()