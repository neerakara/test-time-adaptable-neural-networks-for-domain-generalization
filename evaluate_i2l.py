import os
import logging
import numpy as np
import tensorflow as tf
import utils
import model as model

import data.data_hcp as data_hcp
import data.data_abide as data_abide

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

import sklearn.metrics as met
import config.system as sys_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from experiments import i2l as exp_config

# ============================================================================
# def compute_and_save_results
# ============================================================================
def compute_and_save_results(images,
                             labels,
                             patnames,
                             session,
                             results_fname):
    images_dice = []

    # ================================   
    # open a text file for writing the mean dice scores for each subject that is evaluated
    # ================================   
    results_file = open(os.path.join(sys_config.log_root, exp_config.experiment_name_i2l + '/results/' + results_fname + '.txt'),"w")
    results_file.write("================================== \n") 
    results_file.write("Test results \n") 
    results_file.write("Patient id: mean, std. deviation over foreground labels\n")
    
    # ================================        
    # go through one subject at a time
    # ================================
    num_subjects = images.shape[0] // exp_config.image_depth

    for subject_num in range(num_subjects):

        # ================================
        # print the current subject number to indicate progress
        # ================================
        if subject_num % 1 is 0:
            logging.info('Subject number: %d / %d ... ' % (subject_num, num_subjects))
        
        # ================================
        # extract a subject's image
        # ================================
        image = images[subject_num*exp_config.image_depth : (subject_num+1)*exp_config.image_depth, :, :]
        label = labels[subject_num*exp_config.image_depth : (subject_num+1)*exp_config.image_depth, :, :]
        patname = patnames[subject_num]
        
        # ================================
        # initialize a list for saving the network outputs
        # ================================
        mask_predicted = []
        
        # ================================
        # divide the images into batches
        # ================================
        for b_i in range(0, image.shape[0], batch_size_test):
        
            # ================================            
            # extract the image of this subject and reshape it as required by the network
            # ================================
            X = np.expand_dims(image[b_i:b_i+batch_size_test, ...], axis=-1)
            
            # ================================
            # get the prediction for this batch from the network
            # ================================
            mask_predicted.append(sess.run(mask,  feed_dict={images_pl: X}))
        
        # ================================
        # ================================
        mask_predicted = np.squeeze(np.array(mask_predicted))               
        mask_predicted = mask_predicted.astype(float)
        logging.info('shape of predicted output: %s' %str(mask_predicted.shape))
        
        # ================================
        # swap axes for consistent visualization as for 3d nets
        # ================================
        image = image.swapaxes(1,0)
        label = label.swapaxes(1,0)
        mask_predicted = mask_predicted.swapaxes(1,0)

        # ================================
        # compute the dice for the subject
        # ================================
        # https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        # The function f1_score returns (by setting average to None) the dice scores for each class.
        # ================================
        dice_this_subject = met.f1_score(label.flatten(),
                                         mask_predicted.flatten(),
                                         average=None) 
        images_dice.append(dice_this_subject)
        
        # ================================
        # write the mean dice of this subject to the text file
        # ================================
        results_file.write(patname + ": " + str(np.round(np.mean(dice_this_subject), 3)) + ", " + str(np.round(np.std(dice_this_subject), 3)) + "\n")
                
        # ================================
        # save some qualitative results
        # ================================            
        if subject_num % 1 is 0:
            savepath = os.path.join(sys_config.log_root, exp_config.experiment_name_i2l + '/results/' + results_fname + '_' + patname + '.png')
            utils.save_sample_results1(image,
                                       mask_predicted,
                                       label,
                                       savepath)
            
    # ================================
    # print dice statistics
    # ================================
    images_dice = np.array(images_dice)
    
    # ================================
    # In the array images_dice, in the rows, there are subjects
    # and in the columns, there are the dice scores for each label for a particular subject
    # ================================
    results_file.write("================================== \n") 
    results_file.write("Label: mean, std. deviation over all subjects\n")
    for i in range(images_dice.shape[1]):
        dice_stats = compute_stats(images_dice[:,i])
        logging.info('================================================================')
        logging.info('Dice label %d (mean, median, per5, per95) = %.3f, %.3f, %.3f, %.3f' 
                     % (i, dice_stats[0], dice_stats[1], dice_stats[2], dice_stats[3]))
        results_file.write(str(i) + ": " + str(np.round(np.mean(images_dice[:,i]), 3)) + ", " + str(np.round(np.std(images_dice[:,i]), 3)) + "\n")
    logging.info('================================================================')
    logging.info('Mean dice over all labels: %.3f' % np.mean(images_dice))
    logging.info('Mean dice over all foreground labels: %.3f' % np.mean(images_dice[:,1:]))
    logging.info('================================================================')
    
    # ==================
    # write the mean dice over all subjects and all labels
    # ==================
    results_file.write("================================== \n") 
    results_file.write("Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(images_dice[:,1:]), 3)) + ", " + str(np.round(np.std(images_dice[:,1:]), 3)) + "\n")
    results_file.write("Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(images_dice), 3)) + ", " + str(np.round(np.std(images_dice), 3)) + "\n")
    
    # ==================
    # close the text file
    # ==================
    results_file.write("================================== \n") 
    results_file.close()              

# ============================================================================
# ============================================================================
def compute_stats(array):
    
    mean = np.mean(array)
    median = np.median(array)
    per5 = np.percentile(array,5)
    per95 = np.percentile(array,95)
    
    return mean, median, per5, per95

# ============================================================================
# Main function
# ============================================================================
if __name__ == '__main__':

    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = exp_config.test_dataset
        
    # ===================================
    # read the test images
    # ===================================
    if exp_config.test_dataset is 'HCP_T1':
        imts, gtts, _, pnts  = data_hcp.load_data(sys_config.orig_data_root_hcp,
                                                  sys_config.preproc_folder_hcp,
                                                  'T1w_',
                                                  51,
                                                  71)
        results_file_name = 'hcp_T1w_test_subjects'
        
    elif exp_config.test_dataset is 'HCP_T2':
        imts, gtts, _, pnts  = data_hcp.load_data(sys_config.orig_data_root_hcp,
                                                  sys_config.preproc_folder_hcp,
                                                  'T2w_',
                                                  51,
                                                  71)
        results_file_name = 'hcp_T2w_test_subjects'
    
    # ===================================
    # the resolutions of some datasets may have been modified to match those of the hcp dataset. In such cases, the evaluation is done on these modified resolutions.
    # ===================================
    elif exp_config.test_dataset is 'CALTECH':
        imts, gtts, _, pnts  = data_abide.load_data(input_folder=sys_config.orig_data_root_abide + 'caltech/',
                                                    preproc_folder=sys_config.preproc_folder_abide + 'caltech/',
                                                    idx_start=18,
                                                    idx_end=36,
                                                    bias_correction=True) # test
        results_file_name = 'caltech_test_subjects_bias_corrected'
    
    # ====================================
    # placeholders for images and ground truth labels
    # ====================================    
    nx, ny = exp_config.image_size[:2]
    batch_size_test = exp_config.batch_size_test
    num_channels = exp_config.nlabels
    image_tensor_shape = [batch_size_test] + list(exp_config.image_size) + [1]
    labels_tensor_shape = [None] + list(exp_config.image_size)
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    labels_pl = tf.placeholder(tf.uint8, shape=labels_tensor_shape, name='labels')
    
    # ====================================
    # create predict ops
    # ====================================        
    logits, softmax, mask = model.predict_i2l(images_pl,
                                              exp_config,
                                              training_pl = tf.constant(False, dtype=tf.bool))
    
    # ====================================
    # saver instance for loading the trained parameters
    # ====================================
    saver = tf.train.Saver()
    
    # ====================================
    # add initializer Ops
    # ====================================
    logging.info('Adding the op to initialize variables...')
    init_ops = tf.global_variables_initializer()
    
    # ================================================================
    # freeze the graph before execution
    # ================================================================
    logging.info('Freezing the graph now!')
    tf.get_default_graph().finalize()
    
    with tf.Session() as sess:
        
        # ====================================
        # Initialize
        # ====================================
        sess.run(init_ops)

        # ====================================
        # get the log-directory. the trained models will be saved here.
        # ====================================
        path_to_model = os.path.join(sys_config.log_root, exp_config.experiment_name_i2l + '/models/')
        logging.info('========================================================')
        logging.info('Model directory: %s' % path_to_model)
        
        # ====================================
        # load the model
        # ====================================
        if exp_config.load_this_iter == 0:
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        else:
            checkpoint_path = os.path.join(path_to_model, 'model.ckpt-%d' % exp_config.load_this_iter)
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver.restore(sess, checkpoint_path)
        
        # ====================================
        # create a results directory is it does not exist
        # ====================================
        results_dir = os.path.join(sys_config.log_root, exp_config.experiment_name_i2l + '/results')
        if not tf.gfile.Exists(results_dir):
            tf.gfile.MakeDirs(results_dir)
    
        # ====================================
        # evaluate the test images using the graph parameters           
        # ====================================
        compute_and_save_results(imts,
                                 gtts,
                                 pnts,
                                 sess,
                                 results_file_name)        