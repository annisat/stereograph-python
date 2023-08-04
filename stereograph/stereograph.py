##### HOW TO USE #####
# 1. Prepare an image (e.g. my_image.png). Don't use lossy image (most commonly .jpg).
# 2. Copy your image and rename them for masks (e.g. my_image_mask1.png, my_image_mask2.png)
# 3. Open mask files (e.g. my_image_mask2.png)  with MichaelSoft Paint.
# 4. Use Polygon -> Filled -> Set silhouette color and fill color to, e.g. (2, 2, 2). 2 is the "protruding height" of your stereograph.
# 5. `python stereograph.py make_filter my_image_mask2.png 2`
# 6. `python stereograph.py merge_filters main_filter.png my_image_mask1.png my_image_mask2.png ...`
# 7. `python stereograph.py do my_image.png main_filter.png`
# If the size of `my_image.png` is too large for visions to overlap, set resizing factor to adjust image.
# e.g. `python stereograph.py do my_image.png main_filter.png --ratio=0.5`

import numpy as np
import cv2, fire



# Input image (fn) and mask (mask) to create stereograph
def do(fn, mask, ratio=1):
    img = cv2.imread(str(fn))
    mask = np.mean(cv2.imread(str(mask)), axis=2, keepdims=True).astype(int)
    img_r = img.copy()
    img_l = img.copy()
    
    for v in np.unique(mask):
        if v == 0: continue
        # if v > 10: continue
        # v *= 2
        s = v//10
        # if s%2:
        # s = v//2+1
        shifted_img = np.pad(img, ((0,0), (0, s), (0,0)), mode='edge')[:, s:, :]
        shifted_mask = np.pad(mask, ((0,0), (0, s), (0,0)), mode='constant', constant_values=0)[:, s:, :]
        img_r = np.where(shifted_mask==v, shifted_img, img_r)
        # else:
            # s = v//2
            # shifted_img = np.pad(img, ((0,0), (s, 0), (0,0)), mode='edge')[:, :-s, :]
            # shifted_mask = np.pad(mask, ((0,0), (s, 0), (0,0)), mode='constant', constant_values=0)[:, :-s, :]
            # img_l = np.where(shifted_mask==v, shifted_img, img_l)
    
    joint_img = np.concatenate([img_l, np.ones((img.shape[0], 20, 3))*255, img_r], axis=1)
    joint_img = np.pad(joint_img, ((50,50), (50,50), (0,0)), mode='constant', constant_values=255)
    joint_img = cv2.resize(joint_img.astype(np.uint8), (int(joint_img.shape[1]*ratio), int(joint_img.shape[0]*ratio)))
    cv2.imwrite('stereo.png', joint_img)
    return None
    
    

# Keep all the gray pixel whose gray value is equal to mask_value
# e.g. mask_value = 10 => Keep all (10, 10, 10) pixels
# Change all other pixels to black (0, 0, 0)
# I use Michaelsoft Paint -> filled polygon to make this mask
def make_filter(mask_fn, mask_value):
    mask_img = cv2.imread(str(mask_fn))
    mask = (mask_img==mask_value).prod(axis=-1)[..., None]
    mask_img = np.where(mask, mask_value, 0)
    cv2.imwrite(str(mask_fn), mask_img.astype(np.uint8))
    return None
    
    

# Read all the masks created by `make_filter` and take their maximum as the final filter
def merge_filters(output_fn, *masks):
    for i, mask in enumerate(masks):
        if not i: final_mask = cv2.imread(str(mask), 0)
        else:
            next_mask = cv2.imread(str(mask), 0)
            final_mask = np.maximum(next_mask, final_mask)
    cv2.imwrite(output_fn, final_mask.astype(np.uint8))
    
    
    
if __name__=="__main__":
    fire.Fire()
        