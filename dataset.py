
import os
import json
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile

def get_dataset(root = 'data'):
    os.makedirs(root, exist_ok=True)
    os.chdir(root)

    # URLs for COCO dataset
    urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        # 'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip'
    }

    # Download and extract each resource
    for name, url in tqdm(urls.items()):
        zip_path = f'{name}.zip'
        urllib.request.urlretrieve(url, zip_path) 
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        os.remove(zip_path)


    
def convert_coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    label_dir = Path(output_dir) / 'labels'
    label_dir.mkdir(exist_ok=True)
    
    for annotation in coco_data['annotations']:
        img_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # [x_min, y_min, width, height]
        
        # Normalize bbox
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        w, h = bbox[2], bbox[3]
        
        # Normalize coordinates (assuming image dimensions are known)
        img_width, img_height = 640, 480  # Replace with actual dimensions
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        # Save label
        label_path = label_dir / f"{img_id}.txt"
        with open(label_path, "a") as f:
            f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

