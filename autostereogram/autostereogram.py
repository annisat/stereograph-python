import cv2, fire
import numpy as np



def horizontal_shift(img, shift):
    shift = int(shift)
    if shift > 0:
        return np.pad(img, ((0,0),(shift,0),(0,0)), mode='constant', constant_values=0)[:, :-shift, :]
    else:
        return np.pad(img, ((0,0),(0,-shift),(0,0)), mode='constant', constant_values=0)[:, -shift:, :]
        
        

# Repeating logo_fn to make a base image
def make_base_image(W, H, logo_fn, outout_fn='base_img.png'):
    logo_img = cv2.imread(logo_fn)
    top, left, even_column = 0, 0, False
    
    base_img = np.ones((H, W, 3))*255
    while left < base_img.shape[1]:
        if even_column: top = -logo_img.shape[0]//2
        else: top = 0
        while top < base_img.shape[0]:
            bottom = min(top+logo_img.shape[0], base_img.shape[0])
            right = min(left+logo_img.shape[1], base_img.shape[1])
            logo_h = bottom-top
            logo_w = right-left
            logo_top = max(0, -top)
            base_top = max(0, top)
            # print(left, right, top, bottom)
            base_img[base_top:bottom, left:right, :] =\
                logo_img[logo_top:logo_h, :logo_w, :]
            top += logo_img.shape[0]
        even_column = not even_column
        left += logo_img.shape[1]
    cv2.imwrite(outout_fn, base_img)
    return None
    
    
    
def make_integrated_mask(mask_fn, logo_fn, output_fn='integrated_mask.png'):
    shift = cv2.imread(logo_fn).shape[1]
    mask_img = cv2.imread(mask_fn)
    left_mask = horizontal_shift(mask_img, -shift).astype(int)
    
    integrated_mask = left_mask.copy() #np.zeros_like(left_mask)
    iter_times = mask_img.shape[1]//shift+1
    for i in range(shift):
        for x in range(left_mask.shape[1]-2*shift):
            integrated_mask[:, x, :] = integrated_mask[:, x+2*shift, :] + left_mask[:, x, :]
    
    cv2.imwrite(output_fn, integrated_mask)
    
    
def do(base_image_fn, integrated_mask_fn, output_fn='autostereogram.png'):
    base_img = cv2.imread(base_image_fn)
    integrated_mask = cv2.imread(integrated_mask_fn)//10
    
    out_img = base_img.copy()
    new_flat = int(np.unique(integrated_mask).mean())
    for v in np.unique(integrated_mask):
        shifted_img = horizontal_shift(base_img, (v-new_flat))
        out_img = np.where(integrated_mask==v, shifted_img, out_img)
    
    cv2.imwrite(output_fn, out_img)
    
    
    
    
    
if __name__=="__main__":
    fire.Fire()