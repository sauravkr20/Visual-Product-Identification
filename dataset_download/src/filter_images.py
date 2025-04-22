import json
import gzip
import csv
import os
import shutil

# Load images metadata mapping image_id -> relative path
image_id_to_path = {}
with gzip.open('../../data/images.csv.gz', 'rt', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row['image_id']
        path = row['path']  # e.g., 'if2e/2e3af037.jpg'
        image_id_to_path[image_id] = path

print(f"Loaded {len(image_id_to_path)} image entries.")

# Load shoe products metadata
with open('shoe_products.json', 'r') as f:
    shoe_products = json.load(f)

# first test set
limited_shoe_products = shoe_products[:1000]

# Collect all unique (image_id, image_path) tuples needed for shoe products
required_images = set()
for product in limited_shoe_products:
    image_ids = []
    if product.get('main_image_id'):
        image_ids.append(product['main_image_id'])
    if product.get('other_image_id'):
        image_ids.extend(product['other_image_id'])
    for img_id in image_ids:
        path = image_id_to_path.get(img_id)
        if path:
            required_images.add((img_id, path, product.get('item_id')))

print(f"Total unique shoe images to download: {len(required_images)}")

image_copy_info = []
total_images = len(required_images)

source_folder = '../../data/all_images/images/small'  # Folder where all images are downloaded already
target_folder = '../../data/shoe_images'       # Folder to copy required images into
os.makedirs(target_folder, exist_ok=True)


for idx, (image_id, image_path, item_id) in enumerate(required_images, start=1):
    print(f"Processing image {idx} of {total_images}: {image_path}")

    src_path = os.path.join(source_folder, image_path)
    dst_path = os.path.join(target_folder, image_path)

    # Ensure target subfolders exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if os.path.exists(src_path):
        if(not os.path.exists(dst_path)):
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {image_path}")
            copied = True
        else:
            print(f"Image already exists: {image_path}")
            copied = True
    else:
        print(f"Missing source image: {image_path}")
        copied = False

    image_copy_info.append({
        "image_id": image_id,
        "image_path": image_path,
        "item_id": item_id,
        "exist": copied 
    })
    print(f"Processed {idx} images.")
    print(f"image_copy_info: {image_id}, {item_id}")

with open('../../data/limited_image_paths.json', 'w') as f:
    json.dump(image_copy_info, f, indent=2)

print(f"Saved copy info for {total_images} images to 'limited_image_paths.json'")


# import boto3
# from botocore import UNSIGNED
# from botocore.config import Config


# Setup S3 client and local folder
# bucket_name = 'amazon-berkeley-objects'
# download_folder = '../data/shoe_images'
# os.makedirs(download_folder, exist_ok=True)

# s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# def download_image(image_relative_path, local_folder, prefix='images/small'):
#     s3_path = f"{prefix}/{image_relative_path}"  # full S3 key
#     local_path = os.path.join(local_folder, image_relative_path)  # preserve folder structure locally
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)

#     if os.path.exists(local_path):
#         print(f"Already downloaded: {local_path}")
#         return True

#     try:
#         s3.download_file(bucket_name, s3_path, local_path)
#         print(f"Downloaded: {local_path}")
#         return True
#     except Exception as e:
#         print(f"Failed to download {local_path}: {e}")
#         return False

# Download images and track status
# image_download_info = []

# for idx, (image_id, image_path) in enumerate(required_images, start=1):
#     print(f"Downloading image {idx} of {total_images}: s3 path: {image_path}")
#     success = download_image(image_path, download_folder, prefix='images/small')

#     image_download_info.append({
#         "image_id": image_id,
#         "image_path": image_path,
#         "downloaded": success
#     })



# Save download info to JSON
# with open('image_paths.json', 'w') as f:
#     json.dump(image_download_info, f, indent=2)

# print(f"Saved download info for {total_images} images to 'image_paths.json'")

# this is fucking slow so I downloaded the all the images from images abo- small and then copied the required imagepaths to another folder and removed other images which were not needed
