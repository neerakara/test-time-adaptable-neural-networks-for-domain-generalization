import numpy as np
import data.data_hcp_3d as data_hcp
import config.system as sys_config
import utils
import utils_vis

target_image_size = (256, 256, 256)
target_resolution = (0.7, 0.7, 0.7)

data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                        preprocessing_folder = sys_config.preproc_folder_hcp,
                                                        idx_start = 0,
                                                        idx_end = 20,             
                                                        protocol = 'T1',
                                                        size = target_image_size,
                                                        depth = 256,
                                                        target_resolution = target_resolution)
labels = data_brain_train['labels']
atlas = []

# ===========================
# ===========================
for subject_num in range(labels.shape[0]):

    label_this_subject = labels[subject_num, ...]

    # visualize the labels    
    utils_vis.save_samples_downsampled(label_this_subject[::8, :, :],
                                       sys_config.preproc_folder_hcp + '/training_image_' + str(subject_num+1) + '_for_making_atlas.png')
    
    # add at least one voxel of each label - so that the 1hot function outputs everything of the same shape
    label_this_subject_ = np.copy(label_this_subject)
    for j in range(15):
        label_this_subject_[j,0,0]=j
    
    label_this_subject_1hot = utils.make_onehot(label_this_subject_)
    atlas.append(label_this_subject_1hot)
    
# ===========================
# ===========================
atlas_mean = np.mean(np.array(atlas), axis=0)
atlas_mean = atlas_mean.astype(np.float16)
np.save(sys_config.preproc_folder_hcp + 'hcp_atlas.npy', atlas_mean)
    
atlas_mean_vis = (255*atlas_mean).astype(np.uint8)
for l in range(atlas_mean_vis.shape[-1]):
    utils_vis.save_samples_downsampled(atlas_mean_vis[::8, :, :, l],
                                   sys_config.preproc_folder_hcp + '/hcp_atlas_label' + str(l)+'.png',
                                   add_pixel_each_label=False,
                                   cmap='gray')    
## ===========================
## ===========================    
atlas_hard = np.argmax(atlas_mean_vis, axis=-1)
utils_vis.save_samples_downsampled(atlas_hard[::8, :, :],
                               sys_config.preproc_folder_hcp + '/hcp_atlas_hard.png')