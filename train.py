

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb
import cv2

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from transformers import SamProcessor
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import monai
from scipy.ndimage import zoom
import random 

from transformers import SamModel 

INPUT_PATCH_SIZE = 1024
num_epochs = 10
image_dir = '../data/data_crop1024_shift512/train_images'
mask_dir = '../data/data_crop1024_shift512/train_mask'
positive_file = '../data/positive_list.txt'
negative_file = '../data/negative_list.txt'
hard_negative_file = '../data/hard_negative_samples.txt'
checkpoint_dir = 'checkpoints_valid/'

with open(positive_file, 'r') as f:
    positive_list = f.readlines()

with open(negative_file, 'r') as f:
    negative_list = f.readlines()

with open(hard_negative_file, 'r') as f:
    hard_negative_list = f.readlines()

class SAMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, transform = None):
        self.processor = processor

        # get mask file path list

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        self.mask_path_list = os.listdir(mask_dir)
        self.transform = transform

        self.positive_list = positive_list 
        self.negative_list = negative_list
        self.hard_negative_list = hard_negative_list
        
    
    def __len__(self):
        return len(self.positive_list) * 2
    
    def __getitem__(self, idx):
        

        if random.random() > 0.5: 
            # select postive 
            cur_filename = random.choice(self.positive_list).strip()
        else:
            if random.random() > 0.5: 
                # select random negative
                cur_filename = random.choice(self.negative_list).strip()
            else:
                # select hard negative
                cur_filename = random.choice(self.hard_negative_list).strip()


        mask_path = os.path.join(self.mask_dir,cur_filename)
        # mask = Image.open(mask_path)
        # mask = mask.resize((256,256))
        # mask = np.array(mask)
        # mask[mask==2] =  1

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask, (256, 256))
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1].astype(bool)

        img_path = os.path.join(self.img_dir, cur_filename)
        image = Image.open(img_path)


        if self.transform:
            image, mask = self.transform(image, mask)

        mask = zoom(mask, 256./INPUT_PATCH_SIZE, order=1)  # order=1 for bilinear interpolation

        # prompt = self.get_bounding_box(mask)
        prompt = [0,0,INPUT_PATCH_SIZE,INPUT_PATCH_SIZE]
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        
        # pdb.set_trace()
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        
        # add ground truth segmentation
        inputs["ground_truth_mask"] = mask
        
        return inputs

    
# Define transformations for both images and masks using torchvision.transforms.v2 and RandAugment
transform = T.Compose([
    T.RandomResizedCrop((1024, 1024), scale=(0.8, 1.2)),  # Random resized crop
    T.RandomHorizontalFlip(),  # Random horizontal flipping
    T.RandomVerticalFlip(),    # Random vertical flipping
    T.RandomRotation(degrees=(-45, 45)),  # Random rotation
    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random shifts (adjust translate values as needed)
])
    

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = SAMDataset(img_dir= image_dir, mask_dir= mask_dir, processor=processor, transform = None)


# Define the sizes for the train and eval sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # Remaining for evaluation

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False)


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
optimizer = Adam(model.mask_decoder.parameters(), lr=5e-6, weight_decay=0)

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
        loop.set_postfix(loss=loss.item())

    val_loss = 0
    # validation loss
    with torch.no_grad(): 
        for batch in val_dataloader:
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)

            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            val_loss += loss.item() * len(batch) 
        average_loss = val_loss/len(val_dataloader) / len(val_dataloader.dataset)

        print('validation_loss:', average_loss)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir,'ep'+str(epoch)+'.pth'))


