# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
import config.system as sys_config

import data.data_hcp as data_hcp
import data.data_abide as data_abide

from skimage.transform import rescale

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
    
# ==================================================================
# main function for training
# ==================================================================
def predict_segmentation(subject_name,
                         image,
                         normalize = True):
    
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32,
                                   shape = [None] + list(exp_config.image_size) + [1],
                                   name = 'images')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl,
                                                            exp_config,
                                                            training_pl = tf.constant(False, dtype=tf.bool))
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        predicted_seg_logits, predicted_seg_softmax, predicted_seg = model.predict_i2l(images_normalized,
                                                                                       exp_config,
                                                                                       training_pl = tf.constant(False, dtype=tf.bool))
                        
        # ================================================================
        # 3d prior
        # ================================================================
        labels_3d_1hot_shape = [1] + list(exp_config.image_size_downsampled) + [exp_config.nlabels]
        # predict the current segmentation for the entire volume, downsample it and pass it through this placeholder
        predicted_seg_1hot_3d_pl = tf.placeholder(tf.float32, shape = labels_3d_1hot_shape, name = 'predicted_labels_3d')
        
        # denoise the noisy segmentation
        _, predicted_seg_softmax_3d_noisy_autoencoded_softmax, _ = model.predict_l2l(predicted_seg_1hot_3d_pl,
                                                                                     exp_config,
                                                                                     training_pl = tf.constant(False, dtype=tf.bool))
                
        # ================================================================
        # divide the vars into segmentation network and normalization network
        # ================================================================
        i2l_vars = []
        l2l_vars = []
        normalization_vars = []
        
        for v in tf.global_variables():
            var_name = v.name        
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
                i2l_vars.append(v) # the normalization vars also need to be restored from the pre-trained i2l mapper
            elif 'i2l_mapper' in var_name:
                i2l_vars.append(v)
            elif 'l2l_mapper' in var_name:
                l2l_vars.append(v)
                
                
        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_l2l = tf.train.Saver(var_list = l2l_vars)
        saver_normalizer = tf.train.Saver(var_list = normalization_vars)        
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)
        
        # ================================================================
        # Restore the segmentation network parameters
        # ================================================================ 
        path_to_model = sys_config.log_root + 'i2l_mapper/' + exp_config.expname_i2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the prior network parameters
        # ================================================================
        path_to_model = sys_config.log_root + 'l2l_mapper/' + exp_config.expname_l2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        saver_l2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Make predictions for the image at the resolution of the image after pre-processing
        # ================================================================
        mask_predicted = []
        mask_predicted_soft = []
        img_normalized = []
        
        for b_i in range(0, image.shape[0], 1):
        
            X = np.expand_dims(image[b_i:b_i+1, ...], axis=-1)
            mask_predicted_this_slice = sess.run(predicted_seg, feed_dict={images_pl: X})            
            mask_predicted.append(mask_predicted_this_slice)
            mask_predicted_soft.append(sess.run(predicted_seg_softmax, feed_dict = {images_pl: X}))
            img_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        pred_before_tta = np.squeeze(np.array(mask_predicted)).astype(float)  
        pred_before_tta_soft = np.squeeze(np.array(mask_predicted_soft)).astype(float)  
        imgn_before_tta = np.squeeze(np.array(img_normalized)).astype(float)  
        
        # ================================================================
        # Pass the predictions from the prior network to get the denoised segmentation
        # ================================================================        
        # downsample the predictions
        pred_before_tta_soft_downsampled = rescale(pred_before_tta_soft,
                                                  [1 / exp_config.downsampling_factor_x, 1 / exp_config.downsampling_factor_y, 1 / exp_config.downsampling_factor_z],
                                                  order=1,
                                                  preserve_range=True,
                                                  multichannel=True,
                                                  mode='constant')
        
        feed_dict = {predicted_seg_1hot_3d_pl: np.expand_dims(pred_before_tta_soft_downsampled, axis=0)}                 
        pred_noisy_denoised_softmax = np.squeeze(sess.run(predicted_seg_softmax_3d_noisy_autoencoded_softmax, feed_dict=feed_dict)).astype(np.float16)               
        pred_noisy_denoised = np.argmax(pred_noisy_denoised_softmax, axis=-1)
        
        # upscale the denoised output of the denoising autoencoder
        pred_before_tta_denoised = rescale(pred_noisy_denoised,
                                           [exp_config.downsampling_factor_x, exp_config.downsampling_factor_y, exp_config.downsampling_factor_z],
                                           order=0,
                                           preserve_range=True,
                                           multichannel=False,
                                           mode='constant')
                                                    
        # ================================================================
        # Restore the normalization network parameters (after test time adaptation)
        # ================================================================
        logging.info('============================================================')
        path_to_model = os.path.join(sys_config.log_root, exp_config.expname_normalizer) + '/subject_' + subject_name + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_score.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_normalizer.restore(sess, checkpoint_path)
        logging.info('============================================================')
        
        # ================================================================
        # Make predictions for the image at the resolution of the image after pre-processing
        # ================================================================
        mask_predicted = []
        img_normalized = []
        
        for b_i in range(0, image.shape[0], 1):
        
            X = np.expand_dims(image[b_i:b_i+1, ...], axis=-1)
            mask_predicted_this_slice = sess.run(predicted_seg, feed_dict={images_pl: X})            
            mask_predicted.append(mask_predicted_this_slice)
            img_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        pred_after_tta = np.squeeze(np.array(mask_predicted)).astype(float)  
        imgn_after_tta = np.squeeze(np.array(img_normalized)).astype(float) 
        
        sess.close()
        
        return pred_before_tta, imgn_before_tta, pred_before_tta_denoised, pred_after_tta, imgn_after_tta
        
# ================================================================
# ================================================================
def rescale_and_crop(arr,
                     px,
                     py,
                     nx,
                     ny,
                     order_interpolation,
                     num_rotations):
    
    # 'target_resolution_brain' contains the resolution that the images were rescaled to, during the pre-processing.
    # we need to undo this rescaling before evaluation
    scale_vector = [exp_config.target_resolution_brain[0] / px,
                    exp_config.target_resolution_brain[1] / py]

    arr_list = []
    
    for zz in range(arr.shape[0]):
     
        # ============
        # rotate the labels back to the original orientation
        # ============            
        arr2d_rotated = np.rot90(np.squeeze(arr[zz, :, :]), k=num_rotations)
        
        arr2d_rescaled = rescale(arr2d_rotated,
                                 scale_vector,
                                 order = order_interpolation,
                                 preserve_range = True,
                                 multichannel = False,
                                 mode = 'constant')

        arr2d_rescaled_cropped = utils.crop_or_pad_slice_to_size(arr2d_rescaled, nx, ny)

        arr_list.append(arr2d_rescaled_cropped)
    
    arr_orig_res_and_size = np.array(arr_list)
    arr_orig_res_and_size = arr_orig_res_and_size.swapaxes(0, 1).swapaxes(1, 2)
    
    return arr_orig_res_and_size
        
# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = exp_config.test_dataset
    
    if test_dataset_name is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70       
        
        data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = idx_start,
                                                               idx_end = idx_end,                
                                                               protocol = 'T1',
                                                               size = exp_config.image_size,
                                                               depth = image_depth,
                                                               target_resolution = exp_config.target_resolution_brain)
        
    elif test_dataset_name is 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70
        
        data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = idx_start,
                                                               idx_end = idx_end,           
                                                               protocol = 'T2',
                                                               size = exp_config.image_size,
                                                               depth = image_depth,
                                                               target_resolution = exp_config.target_resolution_brain)
        
    elif test_dataset_name is 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')
        
        image_depth = exp_config.image_depth_caltech
        idx_start = 16
        idx_end = 36         
        
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'CALTECH',
                                                                 idx_start = idx_start,
                                                                 idx_end = idx_end,             
                                                                 protocol = 'T1',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)
        
    imts = data_brain_test['images']
    name_test_subjects = data_brain_test['patnames']
    ids = np.arange(idx_start, idx_end)       
    
    orig_data_res_x = data_brain_test['px'][:]
    orig_data_res_y = data_brain_test['py'][:]
    orig_data_siz_x = data_brain_test['nx'][:]
    orig_data_siz_y = data_brain_test['ny'][:]
    orig_data_siz_z = data_brain_test['nz'][:]
               
    # ================================================================
    # Set subject number here
    # ================================================================
    for sub_num in np.arange(20):
        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject id: %s' %sub_num)
    
        # ==================================================================
        # predict segmentation at the pre-processed resolution
        # ==================================================================
        predicted_labels, normalized_image, denoised_labels, predicted_labels_tta, normalized_image_tta = predict_segmentation(subject_name,
                                                                                                                               image,
                                                                                                                               exp_config.normalize)
    
        # ==================================================================
        # read the original segmentation mask
        # ==================================================================
        if test_dataset_name is 'HCPT1':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                              idx = ids[sub_num],
                                                                              protocol = 'T1',
                                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                              depth = image_depth)
            num_rotations = 0  
            
        elif test_dataset_name is 'HCPT2':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                              idx = ids[sub_num],
                                                                              protocol = 'T2',
                                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                              depth = image_depth)
            num_rotations = 0  
    
        elif test_dataset_name is 'CALTECH':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                               site_name = 'CALTECH',
                                                                               idx = ids[sub_num],
                                                                               depth = image_depth)
            num_rotations = 0
            
        # ==================================================================
        # convert the predicitons back to original resolution
        # ==================================================================
        predicted_labels_orig_res_and_size = rescale_and_crop(predicted_labels,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 0,
                                                              num_rotations = num_rotations)
        
        normalized_image_orig_res_and_size = rescale_and_crop(normalized_image,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 1,
                                                              num_rotations = num_rotations)
        
        denoised_labels_orig_res_and_size = rescale_and_crop(denoised_labels,
                                                             orig_data_res_x[sub_num],
                                                             orig_data_res_y[sub_num],
                                                             orig_data_siz_x[sub_num],
                                                             orig_data_siz_y[sub_num],
                                                             order_interpolation = 0,
                                                             num_rotations = num_rotations)
                                                             
        predicted_labels_tta_orig_res_and_size = rescale_and_crop(predicted_labels_tta,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 0,
                                                              num_rotations = num_rotations)
        
        normalized_image_tta_orig_res_and_size = rescale_and_crop(normalized_image_tta,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 1,
                                                              num_rotations = num_rotations)
            
        # ================================================================
        # save sample results
        # ================================================================
        x_true = image_orig
        y_true = labels_orig
        x_norm = normalized_image_orig_res_and_size
        y_pred = predicted_labels_orig_res_and_size
        y_denoised = denoised_labels_orig_res_and_size
        x_norm_tta = normalized_image_tta_orig_res_and_size
        y_pred_tta = predicted_labels_tta_orig_res_and_size
        
        basepath = os.path.join(sys_config.log_root, exp_config.expname_normalizer) + '/subject_' + subject_name + '/results/'
        for zz in np.arange(80, 120, 10):
            utils_vis.save_single_image(x_true[:,:,zz], basepath + 'slice' + str(zz) + '_x_true.png', 15, False, 'gray', False)
            utils_vis.save_single_image(x_norm[:,:,zz], basepath + 'slice' + str(zz) + '_x_norm.png', 15, False, 'gray', False)
            utils_vis.save_single_image(x_norm_tta[:,:,zz], basepath + 'slice' + str(zz) + '_x_norm_tta.png', 15, False, 'gray', False)
            utils_vis.save_single_image(y_true[:,:,zz], basepath + 'slice' + str(zz) + '_y_true.png', 15, True, 'tab20', False)
            utils_vis.save_single_image(y_pred[:,:,zz], basepath + 'slice' + str(zz) + '_y_pred.png', 15, True, 'tab20', False)
            utils_vis.save_single_image(y_pred_tta[:,:,zz], basepath + 'slice' + str(zz) + '_y_pred_tta.png', 15, True, 'tab20', False)
            utils_vis.save_single_image(y_denoised[:,:,zz], basepath + 'slice' + str(zz) + '_y_pred_dae.png', 15, True, 'tab20', False)
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
