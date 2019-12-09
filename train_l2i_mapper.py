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
import utils_masks

import data.data_hcp as data_hcp

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import l2i as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

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
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 25,             
                                                              protocol = 'T1',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]

        
    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')
                
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
        
        images_pl = tf.placeholder(tf.float32,
                                   shape = [exp_config.batch_size] + list(exp_config.image_size) + [1],
                                   name = 'images')
        
        labels_pl = tf.placeholder(tf.uint8,
                                   shape = [exp_config.batch_size] + list(exp_config.image_size),
                                   name = 'labels')
        
        roi_mask_pl = tf.placeholder(tf.float32,
                                     shape = [exp_config.batch_size] + list(exp_config.image_size) + [1],
                                     name = 'roi_mask')
        
        learning_rate_pl = tf.placeholder(tf.float32,
                                          shape=[],
                                          name = 'learning_rate')
        
        training_pl = tf.placeholder(tf.bool,
                                     shape=[],
                                     name = 'training_or_testing')

        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        pred_images = model.predict_l2i(labels_pl,
                                        exp_config,
                                        training_pl = training_pl)
        
        print('shape of inputs: ', labels_pl.shape) # (batch_size, 256, 256)
        print('shape of pred_images: ', pred_images.shape) # (batch_size, 256, 256, 1)
        
        # ================================================================
        # mask the predicted and ground truth images to remove everything but a box around the labels
        #   - we want to compute the loss only in this region
        #   - there is no way (the rest of the image can be predicted from the labels)
        # ================================================================
        true_images_roi = tf.math.multiply(images_pl,
                                           roi_mask_pl)
        
        pred_images_roi = tf.math.multiply(pred_images,
                                           roi_mask_pl)
        
        # ================================================================
        # create a list of all vars that must be optimized wrt
        # ================================================================
        l2i_vars = []
        for v in tf.trainable_variables():
            l2i_vars.append(v)
        
        # ================================================================
        # add ops for calculation of the supervised training loss
        # ================================================================       
        if exp_config.loss_type_l2i is 'l2':
            loss_op = tf.reduce_mean(tf.square(pred_images_roi - true_images_roi))        
            
        elif exp_config.loss_type_l2i is 'ssim':    
            loss_op = 1 - tf.reduce_mean(tf.image.ssim(img1 = pred_images_roi,
                                                       img2 = true_images_roi,
                                                       max_val = 1.0))
        tf.summary.scalar('loss', loss_op)
        
        # ================================================================
        # add optimization ops.
        # Create different ops according to the variables that must be trained
        # ================================================================
        print('creating training op...')
        train_op = model.training_step(loss_op,
                                       l2i_vars,
                                       exp_config.optimizer_handle,
                                       learning_rate_pl,
                                       update_bn_nontrainable_vars=True)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        print('creating eval op...')
        eval_loss = model.evaluation_l2i(labels = labels_pl,
                                         nlabels = exp_config.nlabels,
                                         predicted_images = pred_images_roi,
                                         true_images = true_images_roi,
                                         loss_type = exp_config.loss_type_l2i,
                                         are_labels_1hot = False)

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
        saver_best_loss = tf.train.Saver(max_to_keep=3)
        
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

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        
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
        curr_lr = exp_config.learning_rate
        best_error = np.inf

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
            for batch in iterate_minibatches(imtr,
                                             gttr,
                                             batch_size = exp_config.batch_size):
                
                curr_lr = exp_config.learning_rate
                start_time = time.time()
                x, y, roi = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue
                
                # ===========================
                # create the feed dict for this training iteration
                # ===========================
                feed_dict = {images_pl: x,
                             labels_pl: y,
                             roi_mask_pl: roi,
                             learning_rate_pl: curr_lr,
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
                    train_loss = do_eval(sess,
                                         eval_loss,
                                         images_pl,
                                         labels_pl,
                                         roi_mask_pl,
                                         training_pl,
                                         imtr,
                                         gttr,
                                         exp_config.batch_size)                    
                    
                    tr_summary_msg = sess.run(tr_error_summary, feed_dict={tr_error: train_loss})                    
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
                    val_loss = do_eval(sess,
                                       eval_loss,
                                       images_pl,
                                       labels_pl,
                                       roi_mask_pl,
                                       training_pl,
                                       imvl,
                                       gtvl,
                                       exp_config.batch_size)                    
                    
                    vl_summary_msg = sess.run(vl_error_summary, feed_dict={vl_error: val_loss})                    
                    summary_writer.add_summary(vl_summary_msg, step)

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if best_error > val_loss:
                        best_error = val_loss
                        best_file = os.path.join(log_dir, 'models/best_loss.ckpt')
                        saver_best_loss.save(sess, best_file, global_step=step)
                        logging.info('Found new average best loss on validation sets! - %f -  Saving model.' % val_loss)

                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            regionofinterest_mask_pl,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    loss_ii = 0
    num_batches = 0
    
    for batch in iterate_minibatches(images, labels, batch_size):

        x, y, roi = batch

        if y.shape[0] < batch_size:
            continue
        
        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     regionofinterest_mask_pl: roi,
                     training_time_placeholder: False}
        
        loss = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_ii += loss
        num_batches += 1

    avg_loss = loss_ii / num_batches

    logging.info('  Average segmentation loss: %.4f' % (avg_loss))

    return avg_loss

# ==================================================================
# ==================================================================
def iterate_minibatches(images,
                        labels,
                        batch_size):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue
        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        x = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        
        # ===========================    
        # data augmentation: random elastic deformations, translations, rotations, scaling
        # ===========================      
        x, y = do_data_augmentation(images = x,
                                    labels = y,
                                    data_aug_ratio = exp_config.da_ratio,
                                    sigma = exp_config.sigma,
                                    alpha = exp_config.alpha,
                                    trans_min = exp_config.trans_min,
                                    trans_max = exp_config.trans_max,
                                    rot_min = exp_config.rot_min,
                                    rot_max = exp_config.rot_max,
                                    scale_min = exp_config.scale_min,
                                    scale_max = exp_config.scale_max)  
        
        # ===========================      
        # make a roi mask around the labels - the loss will be computed only within this.
        # ===========================      
        roi_mask = utils_masks.make_roi_mask(labels = y,
                                             roi_type = 'entire_image')

        yield np.expand_dims(x, axis=-1), y, roi_mask

# ==================================================================
# data augmentation: random elastic deformations, translations, rotations, scaling
# ==================================================================        
def do_data_augmentation(images,
                         labels,
                         data_aug_ratio,
                         sigma,
                         alpha,
                         trans_min,
                         trans_max,
                         rot_min,
                         rot_max,
                         scale_min,
                         scale_max):
    
    images_ = np.copy(images)
    labels_ = np.copy(labels)
    
    for i in range(images.shape[0]):
        
        # ========
        # elastic deformation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            images_[i,:,:], labels_[i,:,:] = utils.elastic_transform_image_and_label(images_[i,:,:],
                                                                                     labels_[i,:,:],
                                                                                     sigma = sigma,
                                                                                     alpha = alpha)     
            
        # ========
        # translation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            
            images_[i,:,:] = scipy.ndimage.interpolation.shift(images_[i,:,:],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 1)
            
            labels_[i,:,:] = scipy.ndimage.interpolation.shift(labels_[i,:,:],
                                                               shift = (random_shift_x, random_shift_y),
                                                               order = 0)
            
        # ========
        # rotation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_angle = np.random.uniform(rot_min, rot_max)
            
            images_[i,:,:] = scipy.ndimage.interpolation.rotate(images_[i,:,:],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 1)
            
            labels_[i,:,:] = scipy.ndimage.interpolation.rotate(labels_[i,:,:],
                                                                reshape = False,
                                                                angle = random_angle,
                                                                axes = (1, 0),
                                                                order = 0)
            
        # ========
        # scaling
        # ========
        if np.random.rand() < data_aug_ratio:
            
            n_x, n_y = images_.shape[1], images_.shape[2]
            
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            
            images_i_tmp = transform.rescale(images_[i,:,:], 
                                             scale_val,
                                             order = 1,
                                             preserve_range = True,
                                             mode = 'constant')
            
            
            labels_i_tmp = transform.rescale(labels_[i,:,:],
                                             scale_val,
                                             order = 0,
                                             preserve_range = True,
                                             mode = 'constant')
            
            images_[i,:,:] = utils.crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
            labels_[i,:,:] = utils.crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)
        
    return images_, labels_
        
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