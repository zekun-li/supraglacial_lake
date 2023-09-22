import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import json


    
def gen_poly2mask():
    inputs = "train"
    task = "mask" #prompt #mask
    
    PATH = f"../data/data_crop1024_shift512/{inputs}_images"
    file_list = glob.glob(os.path.join(PATH, "*.jpg"))
    
    id2name = {}
    name2id = {}
    annotation = f"../data/data_crop1024_shift512/{inputs}_poly.json"
    with open(annotation, 'r') as json_file:
        annotation_gt = json.load(json_file)
        for i in range(len(annotation_gt["images"])):
            id2name[annotation_gt["images"][i]["id"]] = annotation_gt["images"][i]["file_name"]
            name2id[annotation_gt["images"][i]["file_name"]] = annotation_gt["images"][i]["id"]
            
    annotation_img_lst = annotation_gt["img2anno"].keys()
    for each in annotation_img_lst:
        gt_mapping = annotation_gt["img2anno"][each]
        image_info = annotation_gt["images"][name2id[each]]
        assert image_info["file_name"] == each, f"File names do not match: {each} vs. {image_info['file_name']}"

        w, h = image_info["width"], image_info["height"]

        img_path = os.path.join(PATH, each)
        image = cv2.imread(img_path)
    
        for i in gt_mapping:
            gt_info = annotation_gt["annotations"][i]
            if task != "mask":
                
                bbox = gt_info["bbox"]
                x, y, w, h = bbox
                color = (0, 255, 0)
                thickness = 2 
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

                if task == "segprompt":
                    # with poly
                    polygon_points = np.array([gt_info["poly"]], dtype=np.int32)
                    if polygon_points.shape[1] != 2:
                        polygon_points = polygon_points.reshape(-1, 2)

                    cv2.fillPoly(image, [polygon_points], 1)
                
            elif task == "mask":
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                polygon_points = np.array([gt_info["poly"]], dtype=np.int32)
                if polygon_points.shape[1] != 2:
                    polygon_points = polygon_points.reshape(-1, 2)

                cv2.fillPoly(binary_mask, [polygon_points], 1)

        name = each.replace(".jpg", "")
        image_save_path = os.path.join(f"{inputs}_{task}", f"{name}_{task}.jpg")
        # print(image_save_path)
        if task != "mask":
            cv2.imwrite(image_save_path, image)
        elif task == "mask":
            # binary_mask_image = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(image_save_path, binary_mask_image)

        
        
if __name__ == '__main__':
    gen_poly2mask()