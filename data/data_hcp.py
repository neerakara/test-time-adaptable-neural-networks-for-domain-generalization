# ==========================
# import modules
# ==========================
import os
import glob
import numpy as np
import logging
# ==========================
# import modules required for reading through layered zip directories
# ==========================
import zipfile, re
# ==========================
# import image and other utility functions
# ==========================
import utils
import config.system as sys_config

# ==========================
# setup logging
# ==========================
logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s')

# ===============================================================
# This function unzips and pre-processes the data if this has not already been done.
# If this already been done, it reads the processed data and returns it.                    
# ===============================================================                         
def load_data(input_folder,
              preproc_folder,
              protocol,
              idx_start,
              idx_end,
              force_overwrite = False):

    '''
    This function is used to load and if necessary preprocess the HCP data
    
    :param input_folder: Folder where the raw HCP challenge data is located 
    :param preproc_folder: Folder where the proprocessed data should be written to
    :param protocol: Can either be 'T1w_', 'T2w_' or 'both'. Indicates the protocol of the training data
    :param data_partition: can be training, validation or testing.
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: return the read data
    '''   
    
    # ==========================
    # create the pre-processing folder, if it does not exist
    # ==========================
    utils.makefolder(preproc_folder)    
    
    logging.info('============================================================')
    logging.info('Loading data for %s images...' % (protocol) )
        
    # ==========================
    # make appropriate filenames according to the requested indices of training, validation and test images
    # ==========================
    config_details = '%sfrom%dto%d_' % (protocol, idx_start, idx_end)
    filepath_images = preproc_folder + config_details + 'images_2d.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15_2d.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    
    # ==========================
    # if the images have not already been extracted, do so
    # ==========================
    if not os.path.exists(filepath_images) or force_overwrite:
        
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        images, masks, affines, patnames = prepare_data(input_folder,
                                                        preproc_folder,
                                                        protocol,
                                                        idx_start,
                                                        idx_end)
        
    else:
        
        logging.info('Already preprocessed this configuration. Loading now...')
        # read from already created npy files
        images = np.load(filepath_images)
        masks = np.load(filepath_masks)
        affines = np.load(filepath_affine)
        patnames = np.load(filepath_patnames)
        
    return images, masks, affines, patnames

# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder,
                 protocol,
                 idx_start,
                 idx_end):
    
    images = []
    affines = []
    patnames = []
    masks = []

    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + '*.zip'))
    logging.info('Number of images in the dataset: %s' % str(len(filenames)))
        
    # ========================
    # iterate through the requested indices
    # ========================
    for idx in range(idx_start, idx_end):
        
        logging.info('============================================================')
        
        # ========================
        # get the file name for this subject
        # ========================
        filename = filenames[idx]
        
        # ========================
        # define how much of the image can be cropped out as it consists of zeros
        # ========================
        x_start = 18; x_end = -18
        y_start = 28; y_end = -27
        z_start = 2; z_end = -34
        # original images are 260 * 311 * 260
        # cropping them down to 224 * 256 * 224
        
        # ========================
        # read the contents inside the top-level subject directory
        # ========================
        with zipfile.ZipFile(filename, 'r') as zfile:
            
            # ========================
            # search for the relevant files
            # ========================
            for name in zfile.namelist():
                
                # ========================
                # search for files inside the T1w directory
                # ========================
                if re.search(r'\/T1w/', name) != None:
                    
                    # ========================
                    # search for .gz files inside the T1w directory
                    # ========================
                    if re.search(r'\.gz$', name) != None:                        

                        # ========================
                        # get the protocol image
                        # ========================
                        if re.search(protocol + 'acpc_dc_restore_brain', name) != None:
                            
                            logging.info('reading image: %s' % name)
                            
                            _filepath = zfile.extract(name, sys_config.preproc_folder_hcp) # extract the image filepath
                            
                            _patname = name[:name.find('/')] # extract the patient name
                            
                            _img_data, _img_affine, _img_header = utils.load_nii(_filepath) # read the 3d image
                            
                            _img_data = _img_data[x_start:x_end,y_start:y_end,z_start:z_end] # discard some pixels as they are always zero.
                            
                            _img_data = utils.normalise_image(_img_data, norm_type='div_by_max') # normalise the image (volume wise)
                            
                            savepath = sys_config.preproc_folder_hcp + _patname + '/preprocessed_image' + protocol + '.nii' # save the pre-processed image
                            utils.save_nii(savepath, _img_data, _img_affine, _img_header)
                            
                            images.append(_img_data) # append to the list of all images, affines and patient names
                            affines.append(_img_affine)
                            patnames.append(_patname)
                            
                        # ========================
                        # get the segmentation mask
                        # ========================
                        if re.search('aparc.aseg', name) != None: # segmentation mask with ~100 classes 
                            
                            if re.search('T1wDividedByT2w_',name) == None:
                                
                                logging.info('reading mask: %s' % name)
                                
                                _segpath = zfile.extract(name, sys_config.preproc_folder_hcp) # extract the segmentation mask
                                
                                _patname = name[:name.find('/')] # extract the patient name
                                
                                _seg_data, _seg_affine, _seg_header = utils.load_nii(_segpath) # read the segmentation mask
                                
                                _seg_data = _seg_data[x_start:x_end,y_start:y_end,z_start:z_end] # discard some pixels as they are always zero.
                                
                                _seg_data = utils.group_segmentation_classes(_seg_data) # group the segmentation classes as required
                                
                                savepath = sys_config.preproc_folder_hcp + _patname + '/preprocessed_gt15.nii' # save the pre-processed segmentation ground truth
                                utils.save_nii(savepath, _seg_data, _seg_affine, _seg_header) 
                                
                                masks.append(_seg_data) # append to the list of all masks

    # ========================
    # convert the lists to arrays
    # ========================
    images = np.array(images)
    affines = np.array(affines)
    patnames = np.array(patnames)
    masks = np.array(masks, dtype = 'uint8')
    
    # ========================
    # merge along the y-zis to get a stack of x-z slices, for the images as well as the masks
    # ========================
    images = images.swapaxes(1,2)
    images = images.reshape(-1,images.shape[2], images.shape[3])
    masks = masks.swapaxes(1,2)
    masks = masks.reshape(-1,masks.shape[2], masks.shape[3])
    
    # ========================
    # save the processed images and masks so that they can be directly read the next time
    # make appropriate filenames according to the requested indices of training, validation and test images
    # ========================
    logging.info('Saving pre-processed files...')
    config_details = '%sfrom%dto%d_' % (protocol, idx_start, idx_end)
    filepath_images = preproc_folder + config_details + 'images_2d.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15_2d.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    np.save(filepath_images, images)
    np.save(filepath_masks, masks)
    np.save(filepath_affine, affines)
    np.save(filepath_patnames, patnames)
                      
    return images, masks, affines, patnames

# ===============================================================
# Main function that runs if this file is run directly
# ===============================================================
if __name__ == '__main__':

    input_folder = sys_config.orig_data_root_hcp
    preproc_folder = sys_config.preproc_folder_hcp

    i, g, _, _ = load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T1w_', 51, 71, force_overwrite=False)      
    
    visualize_images = False
    if visualize_images is True:
        import matplotlib.pyplot as plt
        for z_idx in np.arange(10,120,10):
            plt.figure(figsize=(10,10))
            s_idx = 0
            plt.subplot(121); plt.imshow(i[s_idx,:,z_idx,:], cmap='gray'); plt.title(str(z_idx))
            plt.subplot(122); plt.imshow(g[s_idx,:,z_idx,:], cmap='tab20')
            plt.show()
            plt.close()
    
# ===============================================================
# End of file
# ===============================================================
