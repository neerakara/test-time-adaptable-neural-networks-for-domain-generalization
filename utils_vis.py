import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np

# ===============================================================
# visualization function
# ===============================================================
def save_sample_results1(x,
                         y,
                         gt,
                         savepath):
    
    ids = [16, 48, 80, 112, 144, 176, 224]
    
    y_ = np.copy(y)
    gt_ = np.copy(gt)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_[0,idx,i] = i
            gt_[0,idx,i] = i
        
    plt.figure(figsize=[3*3,3*len(ids)])
    for i in range(len(ids)): 
        plt.subplot(len(ids),3,3*i+1); plt.imshow(x[:,ids[i],:], cmap='gray'); plt.colorbar()
        plt.subplot(len(ids),3,3*i+2); plt.imshow(y_[:,ids[i],:], cmap='tab20'); plt.colorbar(); plt.title('prediction')
        plt.subplot(len(ids),3,3*i+3); plt.imshow(gt_[:,ids[i],:], cmap='tab20'); plt.colorbar(); plt.title('ground truth')
    plt.savefig(savepath)
    plt.close()
    
    
def save_sample_results(x,
                        x_norm,
                        x_diff,
                        y,
                        y_masked,
                        y_pred_cae,
                        gt,
                        x2xnorm2y2xnormhat,
                        x2xnorm2y2xnormhat_minusdeltax,
                        savepath):
    
    ids = [1, 3, 5, 7, 9, 11, 13]
    
    y_ = np.copy(y)
    gt_ = np.copy(gt)
    y_masked_ = np.copy(y_masked)
    y_pred_cae_ = np.copy(y_pred_cae)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_[idx,0,i] = i
            gt_[idx,0,i] = i
            y_masked_[idx,0,i] = i
            y_pred_cae_[idx,0,i] = i
        
    nc = 9
    plt.figure(figsize=[nc*3, 3*len(ids)])
    for i in range(len(ids)): 
        plt.subplot(len(ids), nc, nc*i+1); plt.imshow(x[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('test image')
        plt.subplot(len(ids), nc, nc*i+2); plt.imshow(x_norm[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('normalized image')
        plt.subplot(len(ids), nc, nc*i+3); plt.imshow(x_diff[ids[i],:,:], cmap='gray'); plt.clim([-1.1,1.1]); plt.colorbar(); plt.title('diff')
        plt.subplot(len(ids), nc, nc*i+4); plt.imshow(y_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred')
        plt.subplot(len(ids), nc, nc*i+5); plt.imshow(y_masked_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred masked')
        plt.subplot(len(ids), nc, nc*i+6); plt.imshow(y_pred_cae_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred masked autoencoded')
        plt.subplot(len(ids), nc, nc*i+7); plt.imshow(gt_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('ground truth')
        plt.subplot(len(ids), nc, nc*i+8); plt.imshow(x2xnorm2y2xnormhat[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('label2image')
        plt.subplot(len(ids), nc, nc*i+9); plt.imshow(x2xnorm2y2xnormhat_minusdeltax[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('label2image - deltax')
    plt.savefig(savepath)
    plt.close()
    
def plot_graph(a, b, save_path):
    plt.figure()
    plt.plot(a, b)
    plt.savefig(save_path)
    plt.close()