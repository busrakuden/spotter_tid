import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import sys
sys.path.append('/home/ks0085/Documents/notebooks/butid/helpers/bsldict')
from demo.utils import load_model, prepare_input, load_rgb_video

tid = pd.read_csv('/home/ks0085/Documents/notebooks/butid/dictionary/dictionary.csv', dtype={'video_id': str})
i3d = load_model('/home/ks0085/Documents/notebooks/butid/models/i3d/i3d_mlp.pth.tar', arch='i3d_mlp')

# Device configuration
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
i3d = i3d.to(device)
i3d.eval()

def extract_i3d_features(video_path, num_samples=20, middle_crop=0.1):
    video = load_rgb_video(video_path, fps=25)
    if middle_crop:
        # Find video length and remove 5% from start and end
        video_length = video.shape[1]
        start = int(video_length * middle_crop)
        end = int(video_length * (1 - middle_crop))
        print(f"Cropping video from {video_length} to {end-start} frames")
        video = video[:, start:end, :, :]
        print(f"New video shape: {video.shape}")
        
    video_id = video_path.split('/')[-1].replace('.gif', '')
    input_tensor = prepare_input(video) # Add batch dimension
    samples = []
    for _ in range(num_samples):
        # Uniform sample so that there are 16 frames
        indixes = torch.linspace(0, input_tensor.shape[1] - 1, 16).long()
        input_16 = input_tensor[:, indixes, :, :].unsqueeze(0).to(device)
        samples.append(input_16)
        
    input_16 = torch.cat(samples, dim=0)
    embedding = i3d(input_16)['embds']
    print(f"Extracted I3D features for {video_id}, shape: {embedding.shape}")
    # get mean of embeddings
    embedding = torch.mean(embedding, dim=0)

    return {
        'video_id': video_id,
        'embeddings': embedding.detach().cpu().numpy()
    }
    
    
SOURCE2PATH = { 
    # 'tid_sozluk': '/home/ks0085/Documents/notebooks/butid/dictionary/rgb/tid_sozluk',
    # 'isaretce': '/home/ks0085/Documents/notebooks/butid/dictionary/rgb/isaretce',
    'afid': '/home/ks0085/Documents/notebooks/butid/dictionary/rgb/afid',
}
for source in tid['source'].unique():
    embeds, video_ids, gloss_list = [], [], []
    
    idx = 0
    for _, row in tqdm(tid[tid['source'] == source].groupby('video_id'), desc="Extracting I3D features"):
        try:
            video_id = row['video_id'].values[0]
            gloss = row['gloss'].values[0]
            print(f"Processing video {video_id} with gloss {gloss}")
            base_path = SOURCE2PATH[source]
            video_path = os.path.join(base_path, f'{video_id}.gif')
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video {video_path} not found")
            result = extract_i3d_features(video_path, num_samples=20, middle_crop=0.1)
            embeds.append(result['embeddings'])
            gloss_list.append(gloss)
            video_ids.append(video_id)

            embeddings = np.stack(embeds, axis=0)
            glosses = np.array(gloss_list)
            ids = np.array(video_ids)
            
            os.makedirs('/home/ks0085/Documents/notebooks/butid/dictionary/i3d', exist_ok=True)

            torch.save({
                'embeddings': embeds,
                'glosses': glosses,
                'video_ids': ids
            }, f'/home/ks0085/Documents/notebooks/butid/dictionary/i3d/{source}.i3d.pth')
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")