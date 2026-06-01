import tarfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def plot_crops(tar_path):
    data = defaultdict(dict)
    categories = ['original', 'deepfake', 'face2face', 'faceshifter', 'faceswap', 'neural textures']
    
    with tarfile.open(tar_path, 'r') as tar:
        try:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith('.jpg'):
                    continue
                
                parts = member.name.split('/')
                if len(parts) >= 2:
                    cat = parts[0]
                    video_id = parts[1]
                    
                    if cat in categories and video_id not in data[cat]:
                        f = tar.extractfile(member)
                        if f is not None:
                            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            data[cat][video_id] = img
        except tarfile.ReadError:
            print("Warning: Tar file is incomplete (likely still being written). Reading available images.")
                        
    if 'deepfake' not in data:
        print("No deepfake images extracted. Check the tar file.")
        return

    fake_video_ids = sorted(list(data['deepfake'].keys()))
    
    for fake_vid in fake_video_ids:
        orig_vid = fake_vid.split('_')[0]
        
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(2, 5)
        
        # Row 1: Original (centered)
        if orig_vid in data.get('original', {}):
            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(data['original'][orig_vid])
            ax.set_title(f'Original - {orig_vid}')
            ax.axis('off')
            
        # Row 2: Fakes
        others = ['deepfake', 'face2face', 'faceshifter', 'faceswap', 'neural textures']
        for i, cat in enumerate(others):
            if fake_vid in data.get(cat, {}):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(data[cat][fake_vid])
                ax.set_title(cat.title())
                ax.axis('off')
                
        plt.suptitle(f"Crop Visualization for {fake_vid}")
        plt.tight_layout()
        out_path = f'crop_visualization_{fake_vid}.png'
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {out_path}")

plot_crops('crop_data_visualize/test/shard-000000.tar')
