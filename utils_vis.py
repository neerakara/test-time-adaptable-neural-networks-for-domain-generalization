# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np

# ==========================================================
# ==========================================================
def add_1_pixel_each_class(arr, nlabels=15):
    
    arr_ = np.copy(arr)
    for j in range(nlabels):
        arr_[0,j]=j
    
    return arr_

# ==========================================================
# ==========================================================
def save_single_image(image,
                      savepath,
                      nlabels,
                      add_pixel_each_label=True,
                      cmap='tab20',
                      colorbar=False,
                      climits = []):
        
    plt.figure(figsize=[20,20])            
    
    if add_pixel_each_label:
        image = add_1_pixel_each_class(image, nlabels)
                
    plt.imshow(image, cmap=cmap)
    if climits != []:
        plt.clim([climits[0], climits[1]])
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()

# ==========================================================
# ==========================================================
def save_samples_downsampled(y,
                             savepath,
                             add_pixel_each_label=True,
                             cmap='tab20'):
        
    plt.figure(figsize=[20,10])
    
    for i in range(4):
    
        for j in range(8):
        
            plt.subplot(4, 8, 8*i+j+1)
            
            if add_pixel_each_label:
                labels_this_slice = add_1_pixel_each_class(y[8*i+j,:,:])
            else:
                labels_this_slice = y[8*i+j,:,:]
                
            plt.imshow(labels_this_slice, cmap=cmap)
            plt.colorbar()

    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    

# ==========================================================
# ==========================================================       
def save_sample_prediction_results(x,
                                   x_norm,
                                   y_pred,
                                   gt,
                                   num_rotations,
                                   savepath):
    
    ids = np.arange(48, 256-48, (256-96)//8)
    nc = len(ids)
    nr = 5
    
    y_pred_ = np.copy(y_pred)
    gt_ = np.copy(gt)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_pred_[0,i,idx] = i
            gt_[0,i,idx] = i
            
    # make a binary mask showing locations of incorrect predictions
    incorrect_mask = np.zeros_like(gt_)
    incorrect_mask[np.where(gt_ != y_pred_)] = 1
        
    plt.figure(figsize=[3*nc, 3*nr])
    
    for c in range(nc): 
        
        x_vis = np.rot90(x[:, :, ids[c]], k=num_rotations)
        x_norm_vis = np.rot90(x_norm[:, :, ids[c]], k=num_rotations)
        y_pred_vis = np.rot90(y_pred_[:, :, ids[c]], k=num_rotations)
        gt_vis = np.rot90(gt_[:, :, ids[c]], k=num_rotations)
        incorrect_mask_vis = np.rot90(incorrect_mask[:, :, ids[c]], k=num_rotations)
        
        plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_vis, cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('Image')
        plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(x_norm_vis, cmap='gray'); plt.colorbar(); plt.title('Normalized')
        plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(y_pred_vis, cmap='tab20'); plt.colorbar(); plt.title('Prediction')
        plt.subplot(nr, nc, nc*3 + c + 1); plt.imshow(gt_vis, cmap='tab20'); plt.colorbar(); plt.title('Ground Truth')
        plt.subplot(nr, nc, nc*4 + c + 1); plt.imshow(incorrect_mask_vis, cmap='tab20'); plt.colorbar(); plt.title('Incorrect pixels')
        
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
# ==========================================================
# ==========================================================       
def save_sample_results(x,
                        x_norm,
                        x_diff,
                        y,
                        y_pred_dae,
                        at,
                        gt,
                        savepath):
    
    ids = np.arange(0, x.shape[0], x.shape[0] // 8)
    
    y_ = np.copy(y)
    gt_ = np.copy(gt)
    at_ = np.copy(at)
    y_pred_dae_ = np.copy(y_pred_dae)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_[idx,0,i] = i
            gt_[idx,0,i] = i
            at_[idx,0,i] = i
            y_pred_dae_[idx,0,i] = i
    
    nc = 7
    plt.figure(figsize=[nc*3, 3*len(ids)])
    for i in range(len(ids)): 
        plt.subplot(len(ids), nc, nc*i+1); plt.imshow(x[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('test image')
        plt.subplot(len(ids), nc, nc*i+2); plt.imshow(x_norm[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('normalized image')
        plt.subplot(len(ids), nc, nc*i+3); plt.imshow(x_diff[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('diff')
        plt.subplot(len(ids), nc, nc*i+4); plt.imshow(y_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred')
        plt.subplot(len(ids), nc, nc*i+5); plt.imshow(y_pred_dae_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred denoised')
        plt.subplot(len(ids), nc, nc*i+6); plt.imshow(at_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('atlas')
        plt.subplot(len(ids), nc, nc*i+7); plt.imshow(gt_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('ground truth')
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
def plot_graph(a, b, save_path):
    plt.figure()
    plt.plot(a, b)
    plt.savefig(save_path)
    plt.close()
