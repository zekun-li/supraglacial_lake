#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai


from transformers import SamModel 

INPUT_PATCH_SIZE = 1024
num_epochs = 1
image_dir = '../data/data_crop1024_shift512/train_images'
mask_dir = '../data/data_crop1024_shift512/1024-train-mask-mult'
checkpoint_dir = 'checkpoints/'


class SAMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.processor = processor

        # get mask file path list

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        self.mask_path_list = os.listdir(mask_dir)
        

    def get_bounding_box(self, ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0:
            return [0,0,INPUT_PATCH_SIZE,INPUT_PATCH_SIZE]
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
        return bbox
    
    def __len__(self):
        return len(self.mask_path_list)
    
    def __getitem__(self, idx):
        # item = self.dataset[idx]
        mask_path = os.path.join(self.mask_dir,self.mask_path_list[idx])
        mask = Image.open(mask_path)
        mask = mask.resize((256,256))
        mask = np.array(mask)
        mask[mask==2] =  1

        ground_truth_mask = mask
        img_path = os.path.join(self.img_dir, self.mask_path_list[idx].replace('_mask',''))
        image = Image.open(img_path)
        # image = item["image"]
        # ground_truth_mask = np.array(item["label"])
    
        # get bounding box prompt
        prompt = self.get_bounding_box(ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        
        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        
        return inputs

    
    
    

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(img_dir= image_dir, mask_dir= mask_dir, processor=processor)

train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)


# In[5]:


batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)



model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')




device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    loop = tqdm(train_dataloader) 
    for idx, batch in enumerate(loop):
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.items())



    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir,'ep'+str(epoch)+'.pth'))


