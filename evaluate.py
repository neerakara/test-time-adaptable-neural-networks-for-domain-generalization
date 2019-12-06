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
import sklearn.metrics as met

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
        logits, softmax, preds = model.predict_i2l(images_normalized,
                                                   exp_config,
                                                   training_pl = tf.constant(False, dtype=tf.bool))
                        
        # ================================================================
        # divide the vars into segmentation network and normalization network
        # ================================================================
        i2l_vars = []
        normalization_vars = []
        
        for v in tf.global_variables():
            var_name = v.name        
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
                
                
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
        logging.info('============================================================')        
        path_to_model = sys_config.log_root + 'i2l_mapper/' + exp_config.expname_i2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the normalization network parameters
        # ================================================================
        if normalize is True:
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
            
            mask_predicted.append(sess.run(preds, feed_dict={images_pl: X}))
            img_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        mask_predicted = np.squeeze(np.array(mask_predicted)).astype(float)  
        img_normalized = np.squeeze(np.array(img_normalized)).astype(float)  
        
        sess.close()
        
        return mask_predicted, img_normalized
    
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
                
    elif test_dataset_name is 'STANFORD':
        logging.info('Reading STANFORD images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'STANFORD/')
        
        image_depth = exp_config.image_depth_stanford
        idx_start = 16
        idx_end = 36         
        
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'STANFORD',
                                                                 idx_start = idx_start,
                                                                 idx_end = idx_end,                  
                                                                 protocol = 'T1',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)
        
    imts = data_brain_test['images']
    name_test_subjects = data_brain_test['patnames']
    num_test_subjects = imts.shape[0] // image_depth
    ids = np.arange(idx_start, idx_end)       
    
    orig_data_res_x = data_brain_test['px'][:]
    orig_data_res_y = data_brain_test['py'][:]
    orig_data_res_z = data_brain_test['pz'][:]
    orig_data_siz_x = data_brain_test['nx'][:]
    orig_data_siz_y = data_brain_test['ny'][:]
    orig_data_siz_z = data_brain_test['nz'][:]
            
    # ================================   
    # set the log directory
    # ================================   
    if exp_config.normalize is True:
        log_dir = os.path.join(sys_config.log_root, exp_config.expname_normalizer)
    else:
        log_dir = sys_config.log_root + 'i2l_mapper/' + exp_config.expname_i2l

    # ================================   
    # open a text file for writing the mean dice scores for each subject that is evaluated
    # ================================       
    results_file = open(log_dir + '/' + test_dataset_name + '_' + 'test' + '.txt', "w")
    results_file.write("================================== \n") 
    results_file.write("Test results \n") 
    results_file.write("Patient id: mean, std. deviation over foreground labels (WRT Ground truth labels obtained from Freesurfer)\n")
    
    # ================================================================
    # For each test image, load the best model and compute the dice with this model
    # ================================================================
    dice_per_label_per_subject = []
    hsd_per_label_per_subject = []

    for sub_num in range(num_test_subjects): 

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
        predicted_labels, normalized_image = predict_segmentation(subject_name,
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

        elif test_dataset_name is 'STANFORD':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                               site_name = 'STANFORD',
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
        
        # ==================================================================
        # compute dice at the original resolution
        # ==================================================================    
        dice_per_label_this_subject = met.f1_score(labels_orig.flatten(),
                                                   predicted_labels_orig_res_and_size.flatten(),
                                                   average=None)
        
        # ==================================================================    
        # compute Hausforff distance at the original resolution
        # ==================================================================    
        hsd_per_label_this_subject = utils.compute_surface_distance(y1 = labels_orig,
                                                                    y2 = predicted_labels_orig_res_and_size,
                                                                    nlabels = exp_config.nlabels)
        
        # ================================================================
        # save sample results
        # ================================================================
        utils_vis.save_sample_prediction_results(x = utils.crop_or_pad_volume_to_size_along_z(image_orig, 256),
                                                 x_norm = utils.crop_or_pad_volume_to_size_along_z(normalized_image_orig_res_and_size, 256),
                                                 y_pred = utils.crop_or_pad_volume_to_size_along_z(predicted_labels_orig_res_and_size, 256),
                                                 gt = utils.crop_or_pad_volume_to_size_along_z(labels_orig, 256),
                                                 num_rotations = - num_rotations, # rotate for consistent visualization across datasets
                                                 savepath = log_dir + '/' + test_dataset_name + '_' + 'test' + '_' + subject_name + '.png')
                                   
        # ================================
        # write the mean fg dice of this subject to the text file
        # ================================
        results_file.write(subject_name + ":: dice (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(dice_per_label_this_subject[1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)))
        results_file.write(", hausdorff distance (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(hsd_per_label_this_subject), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)) + "\n")
        
        dice_per_label_per_subject.append(dice_per_label_this_subject)
        hsd_per_label_per_subject.append(hsd_per_label_this_subject)
    
    # ================================================================
    # write per label statistics over all subjects    
    # ================================================================
    dice_per_label_per_subject = np.array(dice_per_label_per_subject)
    hsd_per_label_per_subject =  np.array(hsd_per_label_per_subject)
    
    # ================================
    # In the array images_dice, in the rows, there are subjects
    # and in the columns, there are the dice scores for each label for a particular subject
    # ================================
    results_file.write("================================== \n") 
    results_file.write("Label: dice mean, std. deviation over all subjects\n")
    for i in range(dice_per_label_per_subject.shape[1]):
        results_file.write(str(i) + ": " + str(np.round(np.mean(dice_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,i]), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.write("Label: hausdorff distance mean, std. deviation over all subjects\n")
    for i in range(hsd_per_label_per_subject.shape[1]):
        results_file.write(str(i+1) + ": " + str(np.round(np.mean(hsd_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject[:,i]), 3)) + "\n")
    
    # ==================
    # write the mean dice over all subjects and all labels
    # ==================
    results_file.write("================================== \n") 
    results_file.write("DICE Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(dice_per_label_per_subject[:,1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,1:]), 3)) + "\n")
    results_file.write("HSD Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(hsd_per_label_per_subject), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
