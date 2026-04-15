import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import cv2

# MoviePy Import
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
# Arkadaşlarının verdiği .npy dosyalarının olduğu yer
REPS_DIR = Path(r"C:\Users\busra\Downloads\shubert_reps_vids")

# Orijinal videoların olduğu yer (Kırpma işlemi için gerekli)
ORIGINAL_VIDEOS_DIR = Path(r"C:\Users\busra\Projects\spotter_tid\data\butid\dictionary\rgb\candidate_videos")

# Çıktı Klasörleri
OUTPUT_MP4_DIR = Path("data/butid/dictionary/rgb/shubert_crops_last_layer")
OUTPUT_CSV = "spotting_results_shubert_last_layer.csv"

# Sözlük Dosyası (Yeni oluşturduğun .pth)
DICTIONARY_EMB_PATH = Path(r"C:\Users\busra\Projects\spotter_tid\data\butid\dictionary\shubert\tid_shubert.pth")
DICTIONARY_CSV_PATH = Path(r"C:\Users\busra\Projects\spotter_tid\data\dictionary.csv") 

# Algoritma Ayarları
WINDOW_SIZE = 16 
WINDOW_STRIDE = 4 
FPS = 25
THRESHOLD = 0 # Belirli bir skorun altını kaydetmemek için

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

# ==============================================================================
# 2. YARDIMCI FONKSİYONLAR
# ==============================================================================

def load_shubert_dictionary(dict_path):
    """ .pth dosyasındaki 12 katmanlı veriyi 768 boyutlu prototiplere çevirir. """
    print(f"📚 SHuBERT Sözlüğü yükleniyor: {dict_path}")
    data = torch.load(dict_path, map_location='cpu', weights_only=False)
    
    glosses = np.array(data['glosses'])
    embeddings_list = data['embeddings']
    
    unique_glosses = np.unique(glosses)
    prototype_dict = {}
    
    for gloss in unique_glosses:
        indices = np.where(glosses == gloss)[0]
        gloss_features = []
        for idx in indices:
            feat = embeddings_list[idx]
            if torch.is_tensor(feat): feat = feat.numpy()
            
            # Eğer veri (12, T, 768) ise katman ortalaması al -> (T, 768)
            if feat.ndim == 3: feat = feat[-1]
            
            # Zamansal ortalama al -> (768,)
            gloss_features.append(np.mean(feat, axis=0))
            
        final_proto = np.mean(gloss_features, axis=0)
        prototype_dict[gloss] = final_proto / (np.linalg.norm(final_proto) + 1e-8)
        
    return prototype_dict

def get_spotting_prediction_shubert(reps_path, target_embedding):
    """ .npy dosyasını okur ve kayan pencere ile benzerlik bulur. """
    try:
        # 1. Özellikleri yükle (12, T, 768) veya (T, 768)
        feat = np.load(reps_path)
        if feat.ndim == 3: feat = feat[-1] # Katmanları birleştir
        
        T = feat.shape[0]
        if T < WINDOW_SIZE: return None

        # 2. Sliding Window
        windowed_features = []
        t_mids = []
        for i in range(0, T - WINDOW_SIZE + 1, WINDOW_STRIDE):
            window = feat[i : i + WINDOW_SIZE]
            win_mean = np.mean(window, axis=0)
            windowed_features.append(win_mean / (np.linalg.norm(win_mean) + 1e-8))
            t_mids.append(i + WINDOW_SIZE / 2)
            
        # 3. Benzerlik (Cosine Similarity)
        windowed_features = np.array(windowed_features)
        target = target_embedding.reshape(1, -1)
        dists = pairwise_distances(windowed_features, target, metric="cosine")
        sims = 1 - dists
        
        return {"max_score": np.max(sims), "peak_time": t_mids[np.argmax(sims)] / FPS}
    except Exception as e:
        print(f"Hata ({reps_path.name}): {e}")
        return None

def save_clip_as_mp4(video_path, output_path, center_time, duration_sec=1.2):
    try:
        with VideoFileClip(str(video_path)) as clip:
            start_t = max(0, center_time - (duration_sec / 2))
            end_t = min(clip.duration, center_time + (duration_sec / 2))
            
            # MOVIEPY V2.0+ UYUMU: subclipped kontrolü
            if hasattr(clip, 'subclipped'):
                new_clip = clip.subclipped(start_t, end_t)
            else:
                new_clip = clip.subclip(start_t, end_t)
                
            # Yeniden boyutlandırma (Resized) uyumu
            if hasattr(new_clip, 'resized'):
                new_clip = new_clip.resized(height=224)
            else:
                new_clip = new_clip.resize(height=224)
                
            new_clip.write_videofile(str(output_path), codec='libx264', audio=False, logger=None)
        return True
    except Exception as e:
        print(f"Kırpma Hatası: {e}")
        return False

# ==============================================================================
# 3. ANA DÖNGÜ
# ==============================================================================

# ... (Üst kısımdaki importlar ve fonksiyonlar aynı kalacak) ...

def main():
    OUTPUT_MP4_DIR.mkdir(parents=True, exist_ok=True)

    prototypes = load_shubert_dictionary(DICTIONARY_EMB_PATH)
    
    # CSV'yi sadece root_id temizliği için kullanacağız veya npy ismindeki gloss'a güveneceğiz
    df_csv = pd.read_csv(DICTIONARY_CSV_PATH, dtype={'video_id': str})
    
    # Sadece eldeki .npy dosyalarını tara
    reps_files = list(REPS_DIR.glob("*.npy"))
    
    print(f"🚀 SHuBERT tabanlı spotting başlıyor... ({len(reps_files)} dosya)")
    
    results = []
    for reps_path in tqdm(reps_files):
        # YENİ MANTIK: Dosya adından hem ID hem Gloss çıkaralım
        # Örn: "0012_ANLAMAK_-AA9n..." -> root_id: 0012, gloss_name: ANLAMAK
        parts = reps_path.stem.split('_')
        if len(parts) < 2:
            continue
            
        vid_id = parts[0]
        gloss_name = parts[1].upper() # CSV'deki glosslar genelde büyük harf olur

        # Sözlükte bu kelime var mı?
        if gloss_name not in prototypes:
            # Eğer bulamazsa alternatif olarak CSV'den bakmayı dene (Noktalı yapı için)
            # Bu kısım eşleşme ihtimalini artırır
            continue

        # Spotting Tahmini
        prediction = get_spotting_prediction_shubert(reps_path, prototypes[gloss_name])
        
        # Hata ayıklama için skorları görelim
        if prediction:
            # Skor eşiğini kontrol etmeden önce debug amaçlı print eklenebilir
            # print(f"Analiz: {vid_id} - Skor: {prediction['max_score']}")
            
            if prediction["max_score"] >= THRESHOLD:
                score = prediction["max_score"]
                timestamp = prediction["peak_time"]
                
                # Orijinal Videoyu Bul
                video_path = ORIGINAL_VIDEOS_DIR / f"{reps_path.stem}.mp4"
                
                if video_path.exists():
                    save_name = f"{reps_path.stem}_spot.mp4"
                    save_path = OUTPUT_MP4_DIR / save_name
                    if save_clip_as_mp4(video_path, save_path, timestamp):
                        results.append({
                            "gloss": gloss_name, 
                            "id": vid_id, 
                            "score": round(score, 4), 
                            "timestamp": round(timestamp, 2), 
                            "saved_as": save_name
                        })
                        
                        # Her başarılı işlemde CSV'yi güncelle/yaz
                        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                    else:
                        print(f"❌ Video kesilemedi: {video_path}")
                else:
                    # Video bulunamazsa hata mesajı ver
                    print(f"⚠️ Video dosyası eksik: {video_path}")

    if not results:
        print("\n⚠️ HİÇBİR SONUÇ BULUNAMADI!")
        print("Nedenler: 1. Skorlar THRESHOLD(0.5) altında kalmış olabilir. 2. Dosya yolları/isimleri uyuşmuyor.")
    else:
        print(f"\n✅ İşlem bitti! {len(results)} sonuç '{OUTPUT_CSV}' içine kaydedildi.")

if __name__ == "__main__":
    main()