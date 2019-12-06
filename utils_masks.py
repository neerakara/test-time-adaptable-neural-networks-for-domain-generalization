import numpy as np

# ==================================================================
# ==================================================================        
def make_roi_mask(labels,
                  roi_type = 'fg_only'):
    
    roi_mask = np.zeros_like(labels, dtype = np.float32)

    # ==========================    
    # for each image in the batch
    # ==========================
    for i in range(roi_mask.shape[0]):
        
        # ==========================
        # find the boundbox for this image
        # ==========================
        fg_indices = np.array(np.where(labels[i,:,:] != 0))
        
        # ==========================
        # highlight the fg pixels, if there are fg pixels in this image
        # else highlight the entire image
        # ==========================
        if fg_indices.shape[1] != 0:

            # highlight the bounding box if there are fg pixels in this image            
            # n = 25 # number of extra pixels outside the fg labels
            # roi_mask[i,
            # np.maximum(np.min(fg_indices[0,:])-n, 0) : np.minimum(np.max(fg_indices[0,:])+n, labels.shape[1]),
            # np.maximum(np.min(fg_indices[1,:])-n, 0) : np.minimum(np.max(fg_indices[1,:])+n, labels.shape[2])] = 1.0

            # highlight exactly those pixels which have non-zero label predictions
            roi_mask[i,
                     fg_indices[0,:],
                     fg_indices[1,:]] = 1.0
        else:
             roi_mask[i, :, :] = 1.0
    
    # in this case, provide the roi as the entire image.
    # this is used while training the l2i mapper.
    if roi_type is 'entire_image':
        roi_mask = np.ones_like(labels, dtype = np.float32)
    
    return np.expand_dims(roi_mask, axis=-1)

# ==================================================================
# ==================================================================
def make_noise_masks_2d(shape,
                        mask_type,
                        mask_params,
                        is_num_masks_fixed,
                        is_size_masks_fixed,
                        nlabels,
                        labels_1hot = None):
    
    blank_masks = np.ones(shape = shape)
    wrong_labels = np.zeros(shape = shape)

    # ====================        
    # for each image in the batch
    # ====================
    for i in range(shape[0]):
            
        # ====================
        # make a random number of noise boxes in this image
        # ====================
        if is_num_masks_fixed is True:
            num_noise_squares = mask_params[1]
        else:
            num_noise_squares = np.random.randint(1, mask_params[1]+1)
            
        for _ in range(num_noise_squares):
                
            # ====================
            # choose the size of the noise box randomly 
            # ====================
            if is_size_masks_fixed is True:
                r = mask_params[0]
            else:
                r = np.random.randint(1, mask_params[0]+1)
                
            # ====================
            # choose the center of the noise box randomly 
            # ====================
            mcx = np.random.randint(r+1, shape[1]-r-1)
            mcy = np.random.randint(r+1, shape[2]-r-1)
                
            # ====================
            # set the labels in this box to 0
            # ====================
            blank_masks[i, mcx-r:mcx+r, mcy-r:mcy+r, :] = 0

            if mask_type is 'random':                
                # ====================
                # set the labels in this box to an arbitrary label
                # ====================
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, np.random.randint(nlabels)] = 1
            
            elif mask_type is 'jigsaw':               
                # ====================
                # choose another box in the image from which copy labels to the previous box
                # ====================
                mcx_src = np.random.randint(r+1, shape[1]-r-1)
                mcy_src = np.random.randint(r+1, shape[2]-r-1)
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, :] = labels_1hot[i, mcx_src-r:mcx_src+r, mcy_src-r:mcy_src+r, :]
                
            elif mask_type is 'zeros':
                # ====================                
                # set the labels in this box to zero
                # ====================
                wrong_labels[i, mcx-r:mcx+r, mcy-r:mcy+r, 0] = 1
        
    return blank_masks, wrong_labels

# ==================================================================
# ==================================================================
def make_noise_masks_3d(shape,
                        mask_type,
                        mask_params,
                        nlabels,
                        labels_1hot = None,
                        is_num_masks_fixed = False,
                        is_size_masks_fixed = False):
    
    blank_masks = np.ones(shape = shape)
    wrong_labels = np.zeros(shape = shape)
                   
    # ====================
    # make a random number of noise boxes in this (3d) image
    # ====================
    if is_num_masks_fixed is True:
        num_noise_squares = mask_params[1]
    else:
        num_noise_squares = np.random.randint(1, mask_params[1]+1)
        
    for _ in range(num_noise_squares):
            
        # ====================
        # choose the size of the noise box randomly 
        # ====================
        if is_size_masks_fixed is True:
            r = mask_params[0]
        else:
            r = np.random.randint(1, mask_params[0]+1)
            
        # choose the center of the noise box randomly 
        mcx = np.random.randint(r+1, shape[1]-r-1)
        mcy = np.random.randint(r+1, shape[2]-r-1)
        mcz = np.random.randint(r+1, shape[3]-r-1)
            
        # set the labels in this box to 0
        blank_masks[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, :] = 0
        
        if mask_type is 'squares_jigsaw':               
            # choose another box in the image from which copy labels to the previous box
            mcx_src = np.random.randint(r+1, shape[1]-r-1)
            mcy_src = np.random.randint(r+1, shape[2]-r-1)
            mcz_src = np.random.randint(r+1, shape[3]-r-1)
            wrong_labels[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, :] = labels_1hot[:, mcx_src-r:mcx_src+r, mcy_src-r:mcy_src+r, mcz_src-r:mcz_src+r, :]
            
        elif mask_type is 'squares_zeros':                
            # set the labels in this box to zero
            wrong_labels[:, mcx-r:mcx+r, mcy-r:mcy+r, mcz-r:mcz+r, 0] = 1
    
    return blank_masks, wrong_labels
