import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import math
import cv2
from PIL import Image
from torchvision import transforms

# MoviePy Import (MP4 kaydı için)
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("Hata: MoviePy yüklenemedi.")

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
CROPPED_VIDEOS_DIR = Path(r"C:\Users\busra\Sign_Language_Project\organized_crops\tid_sozluk")

# Çıktı Klasörleri
OUTPUT_MP4_DIR = Path("data/butid/dictionary/rgb/new_additions7_dino_mp4")
OUTPUT_CSV = "spotting_results_8th_videos_dino.csv"

# Sözlük Dosyası (DINO Features)
DICTIONARY_EMB_PATH = Path(r"C:\Users\busra\Projects\spotter_tid\data\tid_dino_features4.pth")
DICTIONARY_CSV_PATH = Path("data/dictionary.csv") 

# DINO Ayarları
MODEL_TYPE = "dinov2_vitl14" # Özellik çıkarırken kullandığın modelle aynı olmalı
BATCH_SIZE = 16 
WINDOW_SIZE = 16  # I3D'deki NUM_IN_FRAMES karşılığı (Kaç karenin ortalaması alınacak)
WINDOW_STRIDE = 4 # Kaydırma adımı
FPS = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# ==============================================================================
# 2. DINO MODEL VE YARDIMCI FONKSİYONLAR
# ==============================================================================

def load_dino_model():
    print(f"🔄 DINOv2 Modeli ({MODEL_TYPE}) yükleniyor...")
    model = torch.hub.load('facebookresearch/dinov2', MODEL_TYPE)
    model.to(DEVICE)
    model.eval()
    return model

def get_dino_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        
        # Logları kapatarak temiz çıktı alıyoruz
        new_clip.write_videofile(str(output_path), codec='libx264', audio=False, fps=FPS, logger=None)
        clip.close()
        return True
    except Exception as e:
        print(f"MP4 Kayıt Hatası ({video_path.name}): {e}")
        return False

def load_dino_dictionary(dict_path):
    """
    DINO özellikleri içeren .pth dosyasını yükler ve prototipleri (ortalama vektör) hesaplar.
    """
    print(f"📚 DINO Sözlüğü yükleniyor: {dict_path}")
    data = torch.load(dict_path, map_location='cpu', weights_only=False)
    
    # Veri yapısı: {'video_ids': [], 'glosses': [], 'embeddings': [Array(T, 1024), ...]}
    glosses = np.array(data['glosses'])
    embeddings_list = data['embeddings'] # Bu bir liste, her eleman farklı boyutta olabilir (T, 1024)
    
    unique_glosses = np.unique(glosses)
    prototype_dict = {}
    
    print("   -> Prototipler hesaplanıyor...")
    for gloss in unique_glosses:
        indices = np.where(glosses == gloss)[0]
        
        gloss_features = []
        for idx in indices:
            # Her bir videonun özelliklerini al (T, 1024)
            feat = embeddings_list[idx]
            # Videonun zamansal ortalamasını al -> (1024,)
            # Bu, videoyu tek bir vektörle temsil eder (Global Average Pooling)
            mean_video_feat = np.mean(feat, axis=0)
            gloss_features.append(mean_video_feat)
            
        # O glossa ait tüm videoların ortalamasını al
        gloss_features = np.array(gloss_features)
        final_prototype = np.mean(gloss_features, axis=0)
        
        # Normalize et (Cosine similarity için önemli)
        final_prototype = final_prototype / np.linalg.norm(final_prototype)
        
        prototype_dict[gloss] = final_prototype
        
    print(f"   -> {len(prototype_dict)} adet DINO prototipi oluşturuldu.")
    return prototype_dict

def extract_features_continuous(model, video_path, transform):
    """
    Sürekli videodan kare kare DINO özellikleri çıkarır.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(transform(pil_img))
    cap.release()
    
    if not frames: return None

    # Batch işleme
    features_list = []
    frame_tensor = torch.stack(frames) # (T, C, H, W)
    
    with torch.no_grad():
        for i in range(0, len(frame_tensor), BATCH_SIZE):
            batch = frame_tensor[i : i + BATCH_SIZE].to(DEVICE)
            output = model(batch)
            features_list.append(output.cpu().numpy())
            
    return np.concatenate(features_list, axis=0) # (Total_Frames, 1024)

def get_spotting_prediction_dino(model, transform, video_path, target_embedding):
    try:
        # 1. Videonun özelliklerini çıkar: (T, 1024)
        video_features = extract_features_continuous(model, video_path, transform)
        if video_features is None: return None
        
        T = video_features.shape[0]
        if T < WINDOW_SIZE: return None # Video pencereden kısaysa işlem yapma

        # 2. Sliding Window ile özellikleri ortala
        # Amacımız: 16 karelik pencerelerin ortalamasını alıp sözlükle kıyaslamak
        windowed_features = []
        t_mids = [] # Her pencerenin orta zamanı (frame index olarak)

        for i in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
            # Pencereyi al: (16, 1024)
            window = video_features[i : i + WINDOW_SIZE]
            # Ortalamasını al: (1024,)
            win_mean = np.mean(window, axis=0)
            windowed_features.append(win_mean)
            # Orta noktayı kaydet
            t_mids.append(i + WINDOW_SIZE / 2)
            
        windowed_features = np.array(windowed_features) # (Num_Windows, 1024)
        
        # 3. Benzerlik Hesabı
        target = target_embedding.reshape(1, -1)
        # Cosine Distance: 0 (Aynı) -> 2 (Zıt). Similarity = 1 - Dist/2 mantığı ya da direkt 1 - distance
        dists = pairwise_distances(windowed_features, target, metric="cosine")
        sims = 1 - dists # (Skor 1'e ne kadar yakınsa o kadar iyi)
        
        max_sim = np.max(sims)
        peak_idx = np.argmax(sims)
        
        peak_frame = t_mids[peak_idx]
        peak_time_sec = peak_frame / FPS
        
        return {"max_score": max_sim, "peak_time": peak_time_sec}

    except Exception as e:
        print(f"Hata ({video_path.name}): {e}")
        return None

# ==============================================================================
# 3. ANA DÖNGÜ
# ==============================================================================

def main():
    if not OUTPUT_MP4_DIR.exists():
        os.makedirs(OUTPUT_MP4_DIR)

    # 1. Modeli Yükle (I3D yerine DINO)
    model = load_dino_model()
    transform = get_dino_transforms()

    # 2. Sözlüğü Yükle
    prototypes = load_dino_dictionary(DICTIONARY_EMB_PATH) 
    gloss_id_map = load_gloss_to_id_map(DICTIONARY_CSV_PATH) 
    
    if not CROPPED_VIDEOS_DIR.exists():
        print(f"HATA: Giriş klasörü bulunamadı -> {CROPPED_VIDEOS_DIR}")
        return

    gloss_folders = [f for f in CROPPED_VIDEOS_DIR.iterdir() if f.is_dir()]
    gloss_folders.sort(key=lambda f: f.name)

    print(f"🚀 Dördüncü video için DINO tabanlı spotting başlıyor (ID <= 300)...")
    
    for folder in tqdm(gloss_folders, desc="İşleniyor"):
        gloss_name = folder.name
        
        if gloss_name not in prototypes: continue
        if gloss_name not in gloss_id_map: continue

        root_id = gloss_id_map[gloss_name] 
        
        # ID <= 300 Filtresi
        try:
            if int(root_id) > 300:
                continue
        except ValueError:
            continue

        target_emb = prototypes[gloss_name]
        
        video_files = list(folder.glob("*.mp4"))
        
        # --- GÜNCELLEME: En az 4 video olmalı ki 4.'yü (index 3) alalım ---
        if len(video_files) < 8:
            continue
        
        video_files.sort(key=lambda f: f.name.lower()) 
        
        # --- GÜNCELLEME: 4. Videoyu al (Index 3) ---
        vid_path = video_files[7] 
        
        # DINO ile Tahmin
        prediction = get_spotting_prediction_dino(model, transform, vid_path, target_emb)
        
        if prediction:
            score = prediction["max_score"]
            timestamp = prediction["peak_time"]
            
            # Eşik değeri (DINO için I3D'den farklı olabilir, sonuçlara göre ayarlayabilirsin)
            # Şimdilik her bulduğunu kaydedip CSV'den elersin.
            
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
                    "note": "5th_video_dino"
                }
                
                df_row = pd.DataFrame([new_row])
                file_exists = os.path.exists(OUTPUT_CSV)
                df_row.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)

    print(f"\n✅ İşlem bitti! Sonuçlar '{OUTPUT_CSV}' dosyasına anlık olarak eklendi.")

if __name__ == "__main__":
    main()