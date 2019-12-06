# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import model as model
import config.system as sys_config
import scipy.ndimage.interpolation
from skimage import transform

import utils
import utils_vis
import utils_masks
import data.data_hcp_3d as data_hcp

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import l2l as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name_l2l)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name_l2l)

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
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    if exp_config.train_dataset is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 20,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth,
                                                                target_resolution = exp_config.target_resolution_brain)
        gttr = data_brain_train['labels']
        
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 25,             
                                                              protocol = 'T1',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth,
                                                              target_resolution = exp_config.target_resolution_brain)
        gtvl = data_brain_val['labels']

    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_subjects, img_size_x, img_size_y, img_size_z]
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')
    
    # visualize downsampled volumes
    for subject_num in range(gttr.shape[0]):
        utils_vis.save_samples_downsampled(gttr[subject_num, ::2, :, :],
                                       savepath = log_dir + '/training_image_' + str(subject_num+1) + '.png')
                
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')        
        true_labels_shape = [exp_config.batch_size] + list(exp_config.image_size)
        true_labels_pl = tf.placeholder(tf.uint8,
                                        shape = true_labels_shape,
                                        name = 'true_labels')
        
        # ================================================================
        # This will be a mask with all zeros in locations of pixels that we want to alter the labels of.
        # Multiply with this mask to have zero vectors for all those pixels.
        # ================================================================        
        blank_masks_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nlabels]
        blank_masks_pl = tf.placeholder(tf.float32,
                                       shape = blank_masks_shape,
                                       name = 'blank_masks')
        
        # ================================================================
        # This will be a mask with all zeros in locations of pixels that we want to alter the labels of.
        # Multiply with this mask to have zero vectors for all those pixels.
        # ================================================================        
        wrong_labels_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nlabels]
        wrong_labels_pl = tf.placeholder(tf.float32,
                                         shape = wrong_labels_shape,
                                         name = 'wrong_labels')
        
        # ================================================================        
        # Training placeholder
        # ================================================================        
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # make true labels 1-hot
        # ================================================================
        true_labels_1hot = tf.one_hot(true_labels_pl, depth = exp_config.nlabels)
        
        # ================================================================
        # Blank certain locations and write wrong labels in those locations
        # ================================================================
        noisy_labels_1hot = tf.math.multiply(true_labels_1hot, blank_masks_pl) + wrong_labels_pl
                
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        autoencoded_logits, _, _ = model.predict_l2l(noisy_labels_1hot,
                                                     exp_config,
                                                     training_pl = training_pl)

        print('shape of input tensor: ', true_labels_pl.shape) # (batch_size, 64, 256, 256)
        print('shape of input tensor converted to 1-hot: ', true_labels_1hot.shape) # (batch_size, 64, 256, 256, 15) 
        print('shape of predicted logits: ', autoencoded_logits.shape) # (batch_size, 64, 256, 256, 15) 
        
        # ================================================================
        # create a list of all vars that must be optimized wrt
        # ================================================================
        l2l_vars = []
        for v in tf.trainable_variables():
            print(v.name)
            l2l_vars.append(v)
        
        # ================================================================
        # add ops for calculation of the supervised training loss
        # ================================================================
        loss_op = model.loss(logits = autoencoded_logits,
                             labels = true_labels_1hot,
                             nlabels = exp_config.nlabels,
                             loss_type = exp_config.loss_type_l2l,
                             are_labels_1hot = True)
        tf.summary.scalar('loss', loss_op)
        
        # ================================================================
        # add optimization ops.
        # Create different ops according to the variables that must be trained
        # ================================================================
        print('creating training op...')
        train_op = model.training_step(loss_op,
                                       l2l_vars,
                                       exp_config.optimizer_handle,
                                       exp_config.learning_rate,
                                       update_bn_nontrainable_vars=True)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        print('creating eval op...')
        eval_loss = model.evaluation_l2l(logits = autoencoded_logits,
                                         labels = true_labels_1hot,
                                         labels_masked = noisy_labels_1hot,
                                         nlabels = exp_config.nlabels,
                                         loss_type = exp_config.loss_type_l2l,
                                         are_labels_1hot = True)

        # ================================================================
        # build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        print('creating summary op...')
        summary = tf.summary.merge_all()

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # ================================================================
        # create saver
        # ================================================================
        saver = tf.train.Saver(max_to_keep=10)
        saver_best_dice = tf.train.Saver(max_to_keep=3)
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])
        
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_ops)

        # ================================================================
        # print names of all variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of all variables:' )
        for v in tf.trainable_variables(): print(v.name)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while (step < exp_config.max_steps):

            if step % 1000 is 0:
                logging.info('============================================================')
                logging.info('step %d' % step)
        
            # ================================================               
            # batches
            # ================================================            
            for batch in iterate_minibatches(gttr, exp_config.batch_size):
                
                start_time = time.time()
                true_labels, blank_masks, wrong_labels = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if true_labels.shape[0] < exp_config.batch_size:
                    step += 1
                    continue
                
                # ===========================
                # create the feed dict for this training iteration
                # ===========================
                feed_dict = {true_labels_pl: true_labels,
                             blank_masks_pl: blank_masks,
                             wrong_labels_pl: wrong_labels, 
                             training_pl: True}
                
                # ===========================
                # opt step
                # ===========================
                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

                # ===========================
                # compute the time for this mini-batch computation
                # ===========================
                duration = time.time() - start_time

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.3f (%.3f sec for the last step)' % (step+1, loss, duration))
                    
                    # ===========================
                    # Update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # Compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:
                    logging.info('Training Data Eval:')
                    train_loss, train_dice = do_eval(sess,
                                                     eval_loss,
                                                     true_labels_pl,
                                                     blank_masks_pl,
                                                     wrong_labels_pl,
                                                     training_pl,
                                                     gttr,
                                                     exp_config.batch_size)                    
                    
                    tr_summary_msg = sess.run(tr_summary,
                                              feed_dict={tr_error: train_loss,
                                                         tr_dice: train_dice})
                    
                    summary_writer.add_summary(tr_summary_msg, step)
                    
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:
                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # Evaluate the model periodically on a validation set 
                # ===========================
                if step % exp_config.val_eval_frequency == 0:
                    logging.info('Validation Data Eval:')
                    val_loss, val_dice = do_eval(sess,
                                                 eval_loss,
                                                 true_labels_pl,
                                                 blank_masks_pl,
                                                 wrong_labels_pl,
                                                 training_pl,
                                                 gtvl,
                                                 exp_config.batch_size)                    
                    
                    vl_summary_msg = sess.run(vl_summary,
                                              feed_dict={vl_error: val_loss,
                                                         vl_dice: val_dice})
                    
                    summary_writer.add_summary(vl_summary_msg, step)

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                        saver_best_dice.save(sess, best_file, global_step=step)
                        logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            true_labels_placeholder,
            blank_masks_placeholder,
            wrong_labels_placeholder,
            training_time_placeholder,
            labels,
            batch_size):

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(labels, batch_size):

        true_labels_eval, blank_masks_eval, wrong_labels_eval = batch

        if true_labels_eval.shape[0] < batch_size:
            continue
        
        feed_dict = {true_labels_placeholder: true_labels_eval,
                     blank_masks_placeholder: blank_masks_eval,
                     wrong_labels_placeholder: wrong_labels_eval,
                     training_time_placeholder: False}
        
        loss, fg_dice = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_ii += loss
        dice_ii += fg_dice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average segmentation loss: %.4f, average dice: %.4f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(labels,
                        batch_size):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_labels = labels.shape[0]
    random_indices = np.random.permutation(n_labels)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(n_labels // batch_size):

        if b_i + batch_size > n_labels:
            continue
        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        
        labels_this_batch = labels[batch_indices, ...]
        
        # ===========================    
        # data augmentation (random elastic transformations)
        # ===========================      
        labels_this_batch = do_data_augmentation(labels = labels_this_batch,
                                                 data_aug_ratio = exp_config.da_ratio,
                                                 sigma = exp_config.sigma,
                                                 alpha = exp_config.alpha,
                                                 trans_min = exp_config.trans_min,
                                                 trans_max = exp_config.trans_max,
                                                 rot_min = exp_config.rot_min,
                                                 rot_max = exp_config.rot_max,
                                                 scale_min = exp_config.scale_min,
                                                 scale_max = exp_config.scale_max)
        
        # ==================    
        # make labels 1-hot
        # ==================
        labels_this_batch_1hot = utils.make_onehot(labels_this_batch, exp_config.nlabels)
                    
        # ===========================      
        # make noise masks that the autoencoder with try to denoise
        # ===========================      
        blank_masks_this_batch, wrong_labels_this_batch = utils_masks.make_noise_masks_3d(shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nlabels],
                                                                                          mask_type = exp_config.mask_type,
                                                                                          mask_params = [exp_config.mask_radius, exp_config.num_squares],
                                                                                          nlabels = exp_config.nlabels,
                                                                                          labels_1hot = labels_this_batch_1hot)

        yield labels_this_batch, blank_masks_this_batch, wrong_labels_this_batch
        
# ==================================================================
#
# ==================================================================        
def do_data_augmentation(labels,
                         data_aug_ratio,
                         sigma,
                         alpha,
                         trans_min,
                         trans_max,
                         rot_min,
                         rot_max,
                         scale_min,
                         scale_max):
    
    labels_ = np.copy(labels[0,...])
        
    # ========
    # elastic deformation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        labels_ = utils.elastic_transform_label_3d(labels_,
                                                   sigma = sigma,
                                                   alpha = alpha)
        
    # ========
    # translation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        random_shift_x = np.random.uniform(trans_min, trans_max)
        random_shift_y = np.random.uniform(trans_min, trans_max)
        
        for zz in range(labels_.shape[0]):
            labels_[zz,:,:] = scipy.ndimage.interpolation.shift(labels_[zz,:,:],
                                                                shift = (random_shift_x, random_shift_y),
                                                                order = 0)
        
    # ========
    # rotation
    # ========
    if np.random.rand() < data_aug_ratio:
        
        random_angle = np.random.uniform(rot_min, rot_max)
        
        for zz in range(labels_.shape[0]):
            labels_[zz,:,:] = scipy.ndimage.interpolation.rotate(labels_[zz,:,:],
                                                            reshape = False,
                                                            angle = random_angle,
                                                            axes = (1, 0),
                                                            order = 0)
            
    # ========
    # scaling
    # ========
    if np.random.rand() < data_aug_ratio:
        
        n_x, n_y = labels_.shape[1], labels_.shape[2]
        
        scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
        
        for zz in range(labels_.shape[0]):
            labels_i_tmp = transform.rescale(labels_[zz,:,:],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             mode = 'constant')
    
            labels_[zz,:,:] = utils.crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)
        
    return np.expand_dims(labels_, axis=0)
        
# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()