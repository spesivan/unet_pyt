import os
import json
import numpy as np # type: ignore
import cv2 # type: ignore

# Dosya yolları
input_dir = 'test'  # Orjinal resimlerin bulunduğu klasör
output_dir = f'output_{input_dir}'  # Çıktı klasörü
images_dir = os.path.join(output_dir, 'image')  # Çıktı resimler klasörü
masks_dir = os.path.join(output_dir, 'mask')  # Çıktı maskeler klasörü

# Klasörleri oluştur
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# JSON dosyasını oku
with open(f'{input_dir}/_annotations.coco.json') as file:
    data = json.load(file)

# Görüntü bilgilerini bir sözlükte sakla
images_info = {item['id']: item for item in data['images']}

# Her annotasyon için maske oluştur
for annotation in data['annotations']:
    image_info = images_info[annotation['image_id']]
    image_path = os.path.join(input_dir, image_info['file_name'])
    mask_path = os.path.join(masks_dir, image_info['file_name'])

    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Maske için boş bir görüntü oluştur
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Poligonları maske olarak çiz
    for seg in annotation['segmentation']:
        poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 255)

    # Maskeyi kaydet
    cv2.imwrite(mask_path, mask)

    # Orijinal resmi yeni klasöre kopyala
    cv2.imwrite(os.path.join(images_dir, image_info['file_name']), image)

print("Resimler ve maskeler oluşturuldu ve kaydedildi.")
