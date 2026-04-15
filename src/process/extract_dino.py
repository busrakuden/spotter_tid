import torch
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append('./')
sys.path.append('./../')

# --- AYARLAR ---
VIDEO_DIR = "../../data/butid/dictionary/rgb/tid_sozluk"          # Videoların olduğu ana klasör
CSV_PATH = "../../data/dictionary.csv"   # dictionary.csv dosyasının yolu
OUTPUT_FILE = "../../data/tid_dino_features4.pth" # Çıktı dosyasının adı
MODEL_TYPE = "dinov2_vitl14"       # Büyük model (1024 boyutlu çıktı verir)
BATCH_SIZE = 16                    # Bellek durumuna göre ayarlayabilirsin
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dino_model(model_type="dinov2_vitl14"):
    print(f"🔄 DINOv2 Modeli ({model_type}) yükleniyor...")
    model = torch.hub.load('facebookresearch/dinov2', model_type)
    model.to(DEVICE)
    model.eval()
    print("✅ Model yüklendi.")
    return model

def get_transforms():
    # DINOv2 için standart ön işleme
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features_from_video(model, video_path, transform):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(transform(pil_img))
    
    cap.release()
    
    if not frames:
        return None

    # Batch işleme
    features_list = []
    frame_tensor = torch.stack(frames) # (T, C, H, W)
    
    with torch.no_grad():
        for i in range(0, len(frame_tensor), BATCH_SIZE):
            batch = frame_tensor[i : i + BATCH_SIZE].to(DEVICE)
            output = model(batch)
            features_list.append(output.cpu().numpy()) # Numpy olarak saklayacağız
            
    return np.concatenate(features_list, axis=0)

def main():
    # Klasör yoksa oluştur (dosya yolu için)
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_dino_model(MODEL_TYPE)
    transform = get_transforms()
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ HATA: CSV bulunamadı -> {CSV_PATH}")
        return

    print(f"📂 CSV okunuyor: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype={'video_id': str})
    
    # Verileri biriktireceğimiz listeler
    dataset_data = {
        "video_ids": [],
        "glosses": [],
        "embeddings": []
    }

    valid_extensions = ['.mp4', '.gif', '.avi', '.mov', '.mkv']
    processed_count = 0

    # DataFrame üzerinde dönüyoruz
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="İşleniyor"):
        video_id = str(row['video_id']).strip()
        gloss = row['gloss']
        
        # Videoyu bul
        video_path = None
        for ext in valid_extensions:
            candidate = Path(VIDEO_DIR) / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        # Eğer ana klasörde yoksa alt klasörlerde ara
        if video_path is None:
            found = []
            for ext in valid_extensions:
                found.extend(list(Path(VIDEO_DIR).rglob(f"{video_id}{ext}")))
            if found:
                video_path = found[0]

        if video_path is None:
            # print(f"⚠️ Video bulunamadı: {video_id}")
            continue

        try:
            # Özellikleri çıkar
            features = extract_features_from_video(model, video_path, transform)
            
            if features is not None:
                # Listelere ekle
                dataset_data["video_ids"].append(video_id)
                dataset_data["glosses"].append(gloss)
                dataset_data["embeddings"].append(features) # Numpy array olarak eklenir
                processed_count += 1
            else:
                print(f"⚠️ Boş video: {video_id}")

        except Exception as e:
            print(f"💥 Hata ({video_id}): {e}")

    # Sonuçları Kaydet
    print(f"\n💾 Toplam {processed_count} video işlendi. Kaydediliyor...")
    torch.save(dataset_data, OUTPUT_FILE)
    print(f"✅ Dosya başarıyla oluşturuldu: {OUTPUT_FILE}")
    
    # Dosya boyutu hakkında bilgi ver
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"📦 Dosya Boyutu: {file_size:.2f} MB")

if __name__ == "__main__":
    main()