# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import shutil
import tensorflow as tf
import numpy as np
import gc
import model as model
import config.system as sys_config
import sklearn.metrics as met
from skimage.transform import rescale
from tfwrapper import losses, layers

import utils
import utils_vis

import data.data_hcp as data_hcp
import data.data_abide as data_abide

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
    
# ==================================================================
# main function for training
# ==================================================================
def run_training(log_dir,
                 image,
                 label,
                 atlas,
                 continue_run,
                 log_dir_first_TD_subject = ''):

    # ============================
    # down sample the atlas - the losses will be evaluated in the downsampled space
    # ============================
    atlas_downsampled = rescale(atlas,
                                [1 / exp_config.downsampling_factor_x, 1 / exp_config.downsampling_factor_y, 1 / exp_config.downsampling_factor_z],
                                order=1,
                                preserve_range=True,
                                multichannel=True,
                                mode='constant')
    atlas_downsampled = utils.crop_or_pad_volume_to_size_along_x_1hot(atlas_downsampled, int(256 / exp_config.downsampling_factor_x))
        
    label_onehot = utils.make_onehot(label, exp_config.nlabels)
    label_onehot_downsampled = rescale(label_onehot,
                                       [1 / exp_config.downsampling_factor_x, 1 / exp_config.downsampling_factor_y, 1 / exp_config.downsampling_factor_z],
                                       order = 1,
                                       preserve_range = True,
                                       multichannel = True,
                                       mode = 'constant')
    label_onehot_downsampled = utils.crop_or_pad_volume_to_size_along_x_1hot(label_onehot_downsampled, int(256 / exp_config.downsampling_factor_x))
    
    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # if continue_run is set to True, load the model parameters saved earlier
    # else start training from scratch
    # ============================
    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1])
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')
        
    # ================================================================
    # reset the graph built so far and build a new TF graph
    # ================================================================
    tf.reset_default_graph()
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders - segmentation net
        # ================================================================
        images_pl = tf.placeholder(tf.float32, shape = [exp_config.batch_size] + list(exp_config.image_size) + [1], name = 'images')        
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl,
                                                            exp_config,
                                                            training_pl)         
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # By setting the 'training_pl' to false directly, the update ops for the moments in the BN layer are not created at all.
        # This allows grouping the update ops together with the optimizer training, while training the normalizer - in case the normalizer has BN.
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
        # add ops for calculation of the prior loss - wrt an atlas or the outputs of the DAE
        # ================================================================
        prior_label_1hot_pl = tf.placeholder(tf.float32,
                                             shape = [exp_config.batch_size_downsampled] + list((exp_config.image_size_downsampled[1], exp_config.image_size_downsampled[2])) + [exp_config.nlabels],
                                             name = 'labels_prior')
        
        # down sample the predicted logits
        predicted_seg_logits_expanded = tf.expand_dims(predicted_seg_logits, axis = 0)
        # the 'upsample' function will actually downsample the predictions, as the scaling factors have been set appropriately
        predicted_seg_logits_downsampled = layers.bilinear_upsample3D_(predicted_seg_logits_expanded,
                                                                       name = 'downsampled_predictions',
                                                                       factor_x = 1 / exp_config.downsampling_factor_x,
                                                                       factor_y = 1 / exp_config.downsampling_factor_y,
                                                                       factor_z = 1 / exp_config.downsampling_factor_z)
        predicted_seg_logits_downsampled = tf.squeeze(predicted_seg_logits_downsampled) # the first axis was added only for the downsampling in 3d
        
        # compute the dice between the predictions and the prior in the downsampled space
        loss_op = model.loss(logits = predicted_seg_logits_downsampled,
                             labels = prior_label_1hot_pl,
                             nlabels = exp_config.nlabels,
                             loss_type = exp_config.loss_type_prior,
                             mask_for_loss_within_mask = None,
                             are_labels_1hot = True)        

        tf.summary.scalar('tr_losses/loss', loss_op) 
                
        # ================================================================  
        # one of the two prior losses will be used in the following manner:
        # the atlas prior will be used when the current prediction is deemed to be very far away from a reasonable solution
        # once a reasonable solution is reached, the dae prior will be used.
        # these 3d computations will be done outside the graph and will be passed via placeholders for logging in tensorboard
        # ================================================================  
        lambda_prior_atlas_pl = tf.placeholder(tf.float32, shape=[], name = 'lambda_prior_atlas')
        lambda_prior_dae_pl = tf.placeholder(tf.float32, shape=[], name = 'lambda_prior_dae')
        tf.summary.scalar('lambdas/prior_atlas', lambda_prior_atlas_pl)
        tf.summary.scalar('lambdas/prior_dae', lambda_prior_dae_pl)
        
        dice3d_prior_atlas_pl = tf.placeholder(tf.float32, shape=[], name = 'dice3d_prior_atlas')
        dice3d_prior_dae_pl = tf.placeholder(tf.float32, shape=[], name = 'dice3d_prior_dae')
        dice3d_gt_pl = tf.placeholder(tf.float32, shape=[], name = 'dice3d_gt')
        tf.summary.scalar('dice3d/prior_atlas', dice3d_prior_atlas_pl)
        tf.summary.scalar('dice3d/prior_dae', dice3d_prior_dae_pl)
        tf.summary.scalar('dice3d/gt', dice3d_gt_pl)
                
        # ================================================================
        # add optimization ops
        # ================================================================
        if exp_config.debug: print('creating training op...')
        
        # create an instance of the required optimizer
        optimizer = exp_config.optimizer_handle(learning_rate = learning_rate_pl)
        
        # initialize variable holding the accumlated gradients and create a zero-initialisation op
        accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in normalization_vars]
        
        # accumulated gradients init op
        accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accumulated_gradients]

        # calculate gradients and define accumulation op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss_op, var_list = normalization_vars)
            # compute_gradients return a list of (gradient, variable) pairs.
        accumulate_gradients_op = [ac.assign_add(gg[0]) for ac, gg in zip(accumulated_gradients, gradients)]

        # define the gradient mean op
        num_accumulation_steps_pl = tf.placeholder(dtype=tf.float32, name='num_accumulation_steps')
        accumulated_gradients_mean_op = [ag.assign(tf.divide(ag, num_accumulation_steps_pl)) for ag in accumulated_gradients]

        # reassemble the gradients in the [value, var] format and do define train op
        final_gradients = [(ag, gg[1]) for ag, gg in zip(accumulated_gradients, gradients)]
        train_op = optimizer.apply_gradients(final_gradients)

        # ================================================================
        # sequence of running opt ops:
        # 1. at the start of each epoch, run accumulated_gradients_zero_op (no need to provide values for any placeholders)
        # 2. in each training iteration, run accumulate_gradients_op with regular feed dict of inputs and outputs
        # 3. at the end of the epoch (after all batches of the volume have been passed), run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
        # 4. finally, run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
        # ================================================================
        # ================================================================
        # previous train_op without accumulation of gradients
        # ================================================================
        # train_op = model.training_step(loss_op, normalization_vars, exp_config.optimizer_handle, learning_rate_pl, update_bn_nontrainable_vars = True)
        
        # ================================================================
        # build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        if exp_config.debug: print('creating summary op...')

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        if exp_config.debug: logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        prior_dae_dice = tf.placeholder(tf.float32, shape=[], name='prior_dae_dice')
        prior_dae_dice_summary = tf.summary.scalar('test_img/prior_dae_dice', prior_dae_dice)
        prior_dae_output_dice_wrt_gt = tf.placeholder(tf.float32, shape=[], name='prior_dae_output_dice_wrt_gt')
        prior_dae_output_dice_wrt_gt_summary = tf.summary.scalar('test_img/prior_dae_output_dice_wrt_gt', prior_dae_output_dice_wrt_gt)        
        prior_atlas_dice = tf.placeholder(tf.float32, shape=[], name='prior_atlas_dice')
        prior_atlas_dice_summary = tf.summary.scalar('test_img/prior_atlas_dice', prior_atlas_dice)
        prior_dae_atlas_dice_ratio = tf.placeholder(tf.float32, shape=[], name='prior_dae_atlas_dice_ratio')
        prior_dae_atlas_dice_ratio_summary = tf.summary.scalar('test_img/prior_dae_atlas_dice_ratio', prior_dae_atlas_dice_ratio)
        prior_dice = tf.placeholder(tf.float32, shape=[], name='prior_dice')
        prior_dice_summary = tf.summary.scalar('test_img/prior_dice', prior_dice)
        gt_dice = tf.placeholder(tf.float32, shape=[], name='gt_dice')
        gt_dice_summary = tf.summary.scalar('test_img/gt_dice', gt_dice)
        
        # ================================================================
        # create savers
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_l2l = tf.train.Saver(var_list = l2l_vars)
        saver_test_data = tf.train.Saver(var_list = normalization_vars, max_to_keep=3)        
        saver_best_loss = tf.train.Saver(var_list = normalization_vars, max_to_keep=3)    
        
        # ================================================================
        # add operations to compute dice between two 3d volumes
        # ================================================================
        pred_3d_1hot_pl = tf.placeholder(tf.float32, shape = list(exp_config.image_size_downsampled) + [exp_config.nlabels], name = 'pred_3d')
        labl_3d_1hot_pl = tf.placeholder(tf.float32, shape = list(exp_config.image_size_downsampled) + [exp_config.nlabels], name = 'labl_3d')
        atls_3d_1hot_pl = tf.placeholder(tf.float32, shape = list(exp_config.image_size_downsampled) + [exp_config.nlabels], name = 'atls_3d')
        
        dice_3d_op_dae = losses.compute_dice_3d_without_batch_axis(prediction = pred_3d_1hot_pl, labels = labl_3d_1hot_pl)
        dice_3d_op_atlas = losses.compute_dice_3d_without_batch_axis(prediction = pred_3d_1hot_pl, labels = atls_3d_1hot_pl)
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('initializing all variables...')
        sess.run(init_ops)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        uninit_variables = sess.run(uninit_vars)
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('This is the list of uninitialized variables:' )
            for v in uninit_variables: print(v)

        # ================================================================
        # Restore the segmentation network parameters and the pre-trained i2i mapper parameters
        # After the adaptation for the 1st TD subject is done, start the adaptation for the subsequent subjects with those parameters
        # ================================================================
        logging.info('============================================================')        
        path_to_model = sys_config.log_root + 'i2l_mapper/' + exp_config.expname_i2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
                
        # ================================================================
        # Restore the prior network parameters
        # ================================================================
        logging.info('============================================================')
        path_to_model = sys_config.log_root + 'l2l_mapper/' + exp_config.expname_l2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_l2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # After the adaptation for the 1st TD subject is done, start the adaptation for the subsequent subjects with those parameters
        # ================================================================
        if log_dir_first_TD_subject is not '':
            logging.info('============================================================')        
            path_to_model = log_dir_first_TD_subject + '/models/'        
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_score.ckpt')        
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
            saver_test_data.restore(sess, checkpoint_path)
            max_steps_tta = exp_config.max_steps
        else:
            max_steps_tta = 5*exp_config.max_steps # run the adaptation of the 1st TD subject for longer
            
        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring normalization module from: %s' %init_checkpoint_path)
            saver_test_data.restore(sess, init_checkpoint_path)
               
        # ================================================================
        # run training epochs
        # ================================================================
        step = init_step
        best_score = 0.0

        while (step < max_steps_tta):
                                
            # ================================================               
            # After every some epochs,
            # Get the prediction for the entire volume, evaluate it using the DAE.
            # Now, decide whether to use the DAE output or the atlas as the ground truth for the next update.
            # ================================================ 
            if (step == init_step) or (step % exp_config.check_ood_frequency is 0): 
                
                # ==================
                # 1. compute the current 3d segmentation prediction                
                # ==================
                y_pred_soft = []
                for batch in iterate_minibatches_images(image,
                                                        batch_size = exp_config.batch_size):
                    y_pred_soft.append(sess.run(predicted_seg_softmax, feed_dict = {images_pl: batch, training_pl: False}))
                y_pred_soft = np.squeeze(np.array(y_pred_soft)).astype(float)  
                y_pred_soft = np.reshape(y_pred_soft, [-1, y_pred_soft.shape[2], y_pred_soft.shape[3], y_pred_soft.shape[4]])
                
                # ==================
                # 2. downsample it. Let's call this guy 'A'              
                # ==================                
                y_pred_soft_downsampled = rescale(y_pred_soft,
                                                  [1 / exp_config.downsampling_factor_x, 1 / exp_config.downsampling_factor_y, 1 / exp_config.downsampling_factor_z],
                                                  order=1,
                                                  preserve_range=True,
                                                  multichannel=True,
                                                  mode='constant').astype(np.float32)
                                
                # ==================
                # 3. pass the downsampled prediction through the DAE and get its output 'B'
                # ==================                
                feed_dict = {predicted_seg_1hot_3d_pl: np.expand_dims(y_pred_soft_downsampled, axis=0)}                 
                y_pred_noisy_denoised_softmax = np.squeeze(sess.run(predicted_seg_softmax_3d_noisy_autoencoded_softmax, feed_dict=feed_dict)).astype(np.float16)               
                y_pred_noisy_denoised = np.argmax(y_pred_noisy_denoised_softmax, axis=-1)
                
                # ==================
                # 4. compute the dice between:
                #       a. 'A' (seg network prediction downsampled) and 'B' (dae network output)
                #       b. 'B' (dae network output) and downsampled gt labels (for debugging, to see if the dae output is close to the gt.)
                #       c. 'A' (seg network prediction downsampled) and 'C' (downsampled atlas)
                # ==================
                dAB = sess.run(dice_3d_op_dae, feed_dict={pred_3d_1hot_pl: y_pred_soft_downsampled, labl_3d_1hot_pl: y_pred_noisy_denoised_softmax})
                dBgt = sess.run(dice_3d_op_dae, feed_dict={pred_3d_1hot_pl:y_pred_noisy_denoised_softmax, labl_3d_1hot_pl:label_onehot_downsampled})
                dAC = sess.run(dice_3d_op_atlas, feed_dict={pred_3d_1hot_pl: y_pred_soft_downsampled, atls_3d_1hot_pl: atlas_downsampled})
                                
                # ==================
                # 5. compute the ratio dice(AB) / dice(AC). pass the ratio through a threshold and decide whether to use the DAE or the atlas as the prior
                # ==================
                ratio_dice = dAB / (dAC + 1e-5)
                # print('ratio of DAE vs atlas: ' + str(ratio_dice))
                if (ratio_dice > exp_config.dae_atlas_ratio_threshold) and (dAC > exp_config.min_atlas_dice):
                    target_labels_for_this_epoch = y_pred_noisy_denoised_softmax
                    prr = dAB
                else:
                    target_labels_for_this_epoch = atlas_downsampled
                    prr = dAC
                    
                # ==================
                # update losses on tensorboard
                # ==================
                summary_writer.add_summary(sess.run(prior_dae_dice_summary, feed_dict={prior_dae_dice: dAB}), step)
                summary_writer.add_summary(sess.run(prior_dae_output_dice_wrt_gt_summary, feed_dict={prior_dae_output_dice_wrt_gt: dBgt}), step)
                summary_writer.add_summary(sess.run(prior_atlas_dice_summary, feed_dict={prior_atlas_dice: dAC}), step)
                summary_writer.add_summary(sess.run(prior_dae_atlas_dice_ratio_summary, feed_dict={prior_dae_atlas_dice_ratio: ratio_dice}), step)
                summary_writer.add_summary(sess.run(prior_dice_summary, feed_dict={prior_dice: prr}), step)
                
                # ==================
                # save best model so far
                # ==================
                if best_score < prr:
                    best_score = prr
                    best_file = os.path.join(log_dir, 'models/best_score.ckpt')
                    saver_best_loss.save(sess, best_file, global_step=step)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_score, step))
                
                # ==================
                # dice wrt gt
                # ==================             
                y_pred = []
                for batch in iterate_minibatches_images(image,
                                                        batch_size = exp_config.batch_size):
                    y_pred.append(sess.run(predicted_seg, feed_dict = {images_pl: batch, training_pl: False}))
                y_pred = np.squeeze(np.array(y_pred)).astype(float)  
                y_pred = np.reshape(y_pred, [-1, y_pred.shape[2], y_pred.shape[3]])
                dice_wrt_gt = met.f1_score(label.flatten(), y_pred.flatten(), average=None) 
                summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice: np.mean(dice_wrt_gt[1:])}), step)
                    
                # ==================
                # visualize results
                # ==================
                if step % exp_config.vis_frequency is 0: 
                    
                    # ===========================
                    # save checkpoint
                    # ===========================
                    logging.info('=============== Saving checkkpoint at step %d ... ' % step)
                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver_test_data.save(sess, checkpoint_file, global_step=step)
                    
                    y_pred_noisy_denoised_upscaled = utils.crop_or_pad_volume_to_size_along_x(rescale(y_pred_noisy_denoised,
                                                                                                      [exp_config.downsampling_factor_x, exp_config.downsampling_factor_y, exp_config.downsampling_factor_z],
                                                                                                      order=0,
                                                                                                      preserve_range=True,
                                                                                                      multichannel=False,
                                                                                                      mode='constant'), image.shape[0]).astype(np.uint8)
                    
                    x_norm = []
                    for batch in iterate_minibatches_images(image,
                                                            batch_size = exp_config.batch_size):
                        x = batch
                        x_norm.append(sess.run(images_normalized, feed_dict = {images_pl: x, training_pl: False}))                        
                    x_norm = np.squeeze(np.array(x_norm)).astype(float)  
                    x_norm = np.reshape(x_norm, [-1, x_norm.shape[2], x_norm.shape[3]])
                    
                    utils_vis.save_sample_results(x = image,
                                                  x_norm = x_norm,
                                                  x_diff = x_norm - image,
                                                  y = y_pred,
                                                  y_pred_dae = y_pred_noisy_denoised_upscaled,
                                                  at = np.argmax(atlas, axis=-1),
                                                  gt = label,
                                                  savepath = log_dir + '/results/visualize_images/step' + str(step) + '.png')

            # ================================================     
            # Part of training ops sequence:
            # 1. At the start of each epoch, run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            # ================================================               
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
                
            # ================================================               
            # batches
            # ================================================    
            for batch in iterate_minibatches_images_and_downsampled_labels(images = image,
                                                                           batch_size = exp_config.batch_size,
                                                                           labels_downsampled = target_labels_for_this_epoch,
                                                                           batch_size_downsampled = exp_config.batch_size_downsampled):

                x, y = batch   
                
                # ===========================
                # define feed dict for this iteration
                # ===========================   
                feed_dict = {images_pl: x,
                             prior_label_1hot_pl: y,
                             learning_rate_pl: exp_config.learning_rate,
                             training_pl: True}
                
                # ================================================     
                # Part of training ops sequence:
                # 2. in each training iteration, run accumulate_gradients_op with regular feed dict of inputs and outputs
                # ================================================               
                sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                num_accumulation_steps = num_accumulation_steps + 1
                
                step += 1
                
            # ================================================     
            # Part of training ops sequence:
            # 3. At the end of the epoch (after all batches of the volume have been passed), run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
            # ================================================     
            sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
                    
            # ================================================================
            # sequence of running opt ops:
            # 4. finally, run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
            # ================================================================    
            sess.run(train_op, feed_dict=feed_dict)            
                
        # ================================================================    
        # ================================================================    
        sess.close()

    # ================================================================      
    # ================================================================    
    gc.collect()
    
    return 0
        
# ==================================================================
# ==================================================================
def iterate_minibatches_images(images,
                               batch_size):
        
    images_ = np.copy(images)
    
    # generate indices to randomly select subjects in each minibatch
    n_images = images_.shape[0]
    random_indices = np.arange(n_images)

    # generate batches in a for loop
    for b_i in range(n_images // batch_size):
        if b_i + batch_size > n_images:
            continue
        batch_indices = random_indices[b_i*batch_size:(b_i+1)*batch_size]
        images_this_batch = np.expand_dims(images_[batch_indices, ...], axis=-1)

        yield images_this_batch
            
# ==================================================================
# ==================================================================
def iterate_minibatches_images_and_downsampled_labels(images,
                                                      batch_size,
                                                      labels_downsampled,
                                                      batch_size_downsampled):
        
    images_ = np.copy(images)
    labels_downsampled_ = np.copy(labels_downsampled)
    
    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images_.shape[0] # 256
    n_labels_downsampled = labels_downsampled_.shape[0] # 64
    random_indices_images = np.arange(n_images)
    random_indices_labels = np.arange(n_labels_downsampled)

    # ===========================
    # generate batches in a for loop
    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue

        images_this_batch = np.expand_dims(images_[random_indices_images[b_i*batch_size:(b_i+1)*batch_size], ...], axis=-1)
        labels_downsampled_this_batch = labels_downsampled_[random_indices_labels[b_i*batch_size_downsampled:(b_i+1)*batch_size_downsampled], ...]

        yield images_this_batch, labels_downsampled_this_batch

# ==================================================================
# ==================================================================
def main(argv):
        
    # ============================
    # Load test image
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')    
    if exp_config.test_dataset is 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        image_depth = exp_config.image_depth_hcp
        data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = 50,
                                                               idx_end = 70,
                                                               protocol = 'T2',
                                                               size = exp_config.image_size,
                                                               depth = image_depth,
                                                               target_resolution = exp_config.target_resolution_brain)
        imts, gtts = [data_brain_test['images'], data_brain_test['labels']]
        num_test_subjects = imts.shape[0] // image_depth
        name_test_subjects = data_brain_test['patnames']
        slice_thickness_in_test_subjects = data_brain_test['pz'][:]
        
    elif exp_config.test_dataset is 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')
        image_depth = exp_config.image_depth_caltech
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'CALTECH',
                                                                 idx_start = 16,
                                                                 idx_end = 36,
                                                                 protocol = 'T1',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)
        
        imts, gtts = [data_brain_test['images'], data_brain_test['labels']]
        num_test_subjects = imts.shape[0] // image_depth
        name_test_subjects = data_brain_test['patnames']
        slice_thickness_in_test_subjects = data_brain_test['pz'][:]
        
    elif exp_config.test_dataset is 'STANFORD':
        logging.info('Reading STANFORD images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'STANFORD/')
        image_depth = exp_config.image_depth_stanford
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'STANFORD',
                                                                 idx_start = 16,
                                                                 idx_end = 36,
                                                                 protocol = 'T1',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)
        
        imts, gtts = [data_brain_test['images'], data_brain_test['labels']]
        num_test_subjects = imts.shape[0] // image_depth
        name_test_subjects = data_brain_test['patnames']
        slice_thickness_in_test_subjects = data_brain_test['pz'][:]
        
    # ================================================================
    # read the atlas
    # ================================================================
    atlas = np.load(sys_config.preproc_folder_hcp + 'hcp_atlas.npy')
    
    # ================================================================
    # create a text file for writing results
    # results of individual subjects will be appended to this file
    # ================================================================
    log_dir_base = os.path.join(sys_config.log_root, exp_config.expname_normalizer)
    if not tf.gfile.Exists(log_dir_base):
        tf.gfile.MakeDirs(log_dir_base)
    
    # ================================================================
    # run the training for each test image
    # ================================================================
    subject_num = int(argv[0])
    for subject_id in range(subject_num, subject_num+1):
        
        subject_id_start_slice = subject_id * image_depth
        subject_id_end_slice = (subject_id + 1) * image_depth
        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        label = gtts[subject_id_start_slice:subject_id_end_slice,:,:] 
        slice_thickness_this_subject = slice_thickness_in_test_subjects[subject_id]
                
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        log_dir_base = os.path.join(sys_config.log_root, exp_config.expname_normalizer)
        subject_name = str(name_test_subjects[subject_id])[2:-1]
        # subject_name = str(subject_id)
        log_dir = log_dir_base + '/subject_' + subject_name
        logging.info('============================================================')
        logging.info('Logging directory: %s' %log_dir)
        logging.info('Subject ID: %d' %subject_id)
        logging.info('Subject name: %s' %subject_name)
        
        # ===========================
        # create dir if it does not exist
        # ===========================
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(log_dir + '/models')
            tf.gfile.MakeDirs(log_dir + '/results')
            tf.gfile.MakeDirs(log_dir + '/results/visualize_images')
            
        # ===========================
        # Copy experiment config file
        # ===========================
        shutil.copy(exp_config.__file__, log_dir)

        # ===========================
        # Change the resolution of the current image so that it matches the atlas, and pad and crop.
        # ===========================
        image_rescaled_cropped, label_rescaled_cropped = modify_image_and_label(image,
                                                                                label,
                                                                                atlas,
                                                                                slice_thickness_this_subject)
        
        # visualize image and ground truth label
        utils_vis.save_samples_downsampled(utils.crop_or_pad_volume_to_size_along_x(image, 256)[::8, :, :], savepath = log_dir + '/orig_image.png', add_pixel_each_label=False, cmap='gray')
        utils_vis.save_samples_downsampled(utils.crop_or_pad_volume_to_size_along_x(label, 256)[::8, :, :], savepath = log_dir + '/gt_label.png', cmap='tab20')
        utils_vis.save_samples_downsampled(image_rescaled_cropped[::8, :, :], savepath = log_dir + '/orig_image_rescaled.png', add_pixel_each_label=False, cmap='gray')
        utils_vis.save_samples_downsampled(label_rescaled_cropped[::8, :, :], savepath = log_dir + '/gt_label_rescaled.png', cmap='tab20')
        
        # ===========================
        # run training. pass the log dir of the 1st TD subject, if this training has been completed successfully
        # ===========================
        first_subject = 0
        log_dir_first_TD_subject = ''
        # if subject_id != first_subject: log_dir_first_TD_subject = log_dir_base + '/subject_' + str(name_test_subjects[first_subject])[2:-1]
        
        run_training(log_dir,
                     image_rescaled_cropped,
                     label_rescaled_cropped,
                     atlas,
                     continue_run = exp_config.continue_run,
                     log_dir_first_TD_subject = log_dir_first_TD_subject)
        
        # ===========================
        # ===========================
        gc.collect()
        
# ===========================================================================
# ===========================================================================
def modify_image_and_label(image,
                           label,
                           atlas,
                           slice_thickness_this_subject):
    
    image_rescaled = []
    label_rescaled = []
            
    # ======================
    # rescale in 3d
    # ======================
    scale_vector = [slice_thickness_this_subject / 0.7, # for this axes, the resolution was kept unchanged during the initial 2D data preprocessing. but for the atlas (made from hcp labels), all of them have 0.7mm slice thickness
                    1.0, # the resolution along these 2 axes was made as required in the initial 2d data processing already
                    1.0]
    
    image_rescaled = rescale(image,
                             scale_vector,
                             order=1,
                             preserve_range=True,
                             multichannel=False,
                             mode = 'constant')

    label_onehot = utils.make_onehot(label, exp_config.nlabels)

    label_onehot_rescaled = rescale(label_onehot,
                                    scale_vector,
                                    order=1,
                                    preserve_range=True,
                                    multichannel=True,
                                    mode='constant')
    
    label_rescaled = np.argmax(label_onehot_rescaled, axis=-1)
        
    # =================
    # crop / pad
    # =================
    image_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(image_rescaled, atlas.shape[0]).astype(np.float32)
    label_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(label_rescaled, atlas.shape[0]).astype(np.uint8)
            
    return image_rescaled_cropped, label_rescaled_cropped

# ==================================================================
# ==================================================================
import sys
if __name__ == "__main__":
    main(sys.argv[1:])
