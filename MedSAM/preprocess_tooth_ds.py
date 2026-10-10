
import json
import os
import cv2
import numpy as np
import shutil
import argparse

def create_binary_dataset(coco_json_path, source_images_dir, output_dir):
    # Create the output folder structure
    images_out_dir = os.path.join(output_dir, "images")
    masks_out_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)
    
    # Load the COCO JSON data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Target category IDs for the unified "disease" class 
    # 1: Caries, 2: Cavity, 3: Crack. (4: Tooth is ignored and left as background)
    disease_category_ids = {1, 2, 3}
    
    # Group annotations by image ID for bulk processing
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
        
    bboxes_json_data = {}
    
    # Process each image
    for img in coco_data.get('images', []):
        image_id = img['id']
        file_name = img['file_name']
        height = img['height']
        width = img['width']
        
        # Use filename without extension as the unique key
        file_key = os.path.splitext(file_name)[0]
        
        # Initialize a black 8-bit binary mask (background)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                # Only draw the mask if it belongs to the disease classes
                if ann['category_id'] in disease_category_ids:
                    if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                        for seg in ann['segmentation']:
                            # Reshape COCO polygon list into (x, y) coordinates
                            poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                            # Fill the polygon with white (255)
                            cv2.fillPoly(mask, [poly], 255)
        
        # Generate new, non-overlapping bounding boxes from the unified mask
        # RETR_EXTERNAL ensures we only get the outermost boundaries, merging overlaps
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        merged_bboxes = []
        for contour in contours:
            # Filter out microscopic noise artifacts (threshold set to 5 pixels)
            if cv2.contourArea(contour) > 5:
                x, y, w, h = cv2.boundingRect(contour)
                merged_bboxes.append({
                    "label": "disease",
                    "xmin": int(x),
                    "ymin": int(y),
                    "xmax": int(x + w),
                    "ymax": int(y + h)
                })
        
        # Construct the JSON structure for this specific image
        bboxes_json_data[file_key] = {
            "height": height,
            "width": width,
            "bbox": merged_bboxes
        }
        
        # Save the binary mask as PNG
        mask_filename = f"{file_key}.png"
        cv2.imwrite(os.path.join(masks_out_dir, mask_filename), mask)
        
        # Route the original image
        src_img_path = os.path.join(source_images_dir, file_name)
        dst_img_path = os.path.join(images_out_dir, f"{file_key}.jpg")
        
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {file_name} not found in {source_images_dir}")

    # Save the central bboxes.json to the root of the output directory
    with open(os.path.join(output_dir, "bboxes.json"), 'w') as f:
        json.dump(bboxes_json_data, f)
        
    print(f"Binary dataset generated. Saved to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON to a binary mask dataset format.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the COCO _annotations.coco.json file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the original JPG images")
    parser.add_argument("--out_dir", type=str, required=True, help="Target output directory for the generated dataset")
    
    args = parser.parse_args()
    
    create_binary_dataset(
        coco_json_path=args.json_path,
        source_images_dir=args.img_dir,
        output_dir=args.out_dir
    )
