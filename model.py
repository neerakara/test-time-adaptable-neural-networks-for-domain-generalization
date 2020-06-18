import tensorflow as tf
from tfwrapper import losses
import matplotlib
import matplotlib.cm

# ================================================================
# ================================================================
def predict_i2l(images,
                exp_config,
                training_pl):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = exp_config.model_handle_i2l(images,
                                         nlabels = exp_config.nlabels,
                                         training_pl = training_pl)
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, softmax, mask

# ================================================================
# ================================================================
def predict_l2i(labels,
                exp_config,
                training_pl):

    labels = tf.one_hot(labels, depth=exp_config.nlabels)
    images = exp_config.model_handle_l2i(labels, training_pl = training_pl)

    return images

# ================================================================
# ================================================================
def predict_l2l(inputs,
                exp_config,
                training_pl):

    logits = exp_config.model_handle_l2l(inputs,
                                         nlabels = exp_config.nlabels,
                                         training_pl = training_pl)
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, softmax, mask

# ================================================================
# ================================================================
def normalize(images,
              exp_config,
              training_pl):
    
    images_normalized, added_residual = exp_config.model_handle_normalizer(images,
                                                                           exp_config,
                                                                           training_pl)
    
    return images_normalized, added_residual
    
# ================================================================
# ================================================================
def loss(logits,
         labels,
         nlabels,
         loss_type,
         mask_for_loss_within_mask = None,
         are_labels_1hot = False):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'crossentropy'/'dice'/
    :return: The segmentation
    '''

    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)

    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
        
    elif loss_type == 'crossentropy_reverse':
        predicted_probabilities = tf.nn.softmax(logits)
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels)
        
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
        
    elif loss_type == 'dice_within_mask':
        if mask_for_loss_within_mask is not None:
            segmentation_loss = losses.dice_loss_within_mask(logits, labels, mask_for_loss_within_mask)

    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    return segmentation_loss

# ================================================================
# ================================================================
def likelihood_loss(pred_img_from_pred_seg_inverted,
                    img_orig,
                    loss_type):
    
    if loss_type is 'l2':
        loss_likelihood_op = tf.reduce_mean(tf.reduce_sum(tf.square(pred_img_from_pred_seg_inverted - img_orig), axis=[1,2,3]))
                
    elif loss_type is 'ssim':    
        loss_likelihood_op = 1 - tf.reduce_mean(tf.image.ssim(img1 = pred_img_from_pred_seg_inverted,
                                                              img2 = img_orig,
                                                              max_val = 1.0))
        
    return loss_likelihood_op

# ================================================================
# ================================================================
def training_step(loss,
                  var_list,
                  optimizer_handle,
                  learning_rate,
                  update_bn_nontrainable_vars,
                  return_optimizer = False):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :var_list: list of params that this loss should be optimized wrt.
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :return: The training operation
    '''

    optimizer = optimizer_handle(learning_rate = learning_rate) 
    train_op = optimizer.minimize(loss, var_list=var_list)
    
    if update_bn_nontrainable_vars is True:
        opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, opt_memory_update_ops])

    if return_optimizer is True:
        return train_op, optimizer
    else:
        return train_op

# ================================================================
# ================================================================
def evaluate_losses(logits,
                    labels,
                    nlabels,
                    loss_type,
                    are_labels_1hot = False):
    '''
    A function to compute various loss measures to compare the predicted and ground truth annotations
    '''
    
    # =================
    # supervised loss that is being optimized
    # =================
    supervised_loss = loss(logits = logits,
                           labels = labels,
                           nlabels = nlabels,
                           loss_type = loss_type,
                           are_labels_1hot = are_labels_1hot)

    # =================
    # per-structure dice for each label
    # =================
    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)
    dice_all_imgs_all_labels, mean_dice, mean_dice_fg = losses.compute_dice(logits, labels)
    
    return supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg

# ================================================================
# ================================================================
def evaluation_i2l(logits,
                   labels,
                   images,
                   nlabels,
                   loss_type):

    # =================
    # compute loss and foreground dice
    # =================
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits,
                                                                                         labels,
                                                                                         nlabels,
                                                                                         loss_type)

    # =================
    # 
    # =================
    mask = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
    mask_gt = labels
    
    # =================
    # write some segmentations to tensorboard
    # =================
    gt1 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=1, nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    pred1 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=1, nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    img1 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=0, nlabels=nlabels)
    img2 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=1, nlabels=nlabels)
    img3 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=2, nlabels=nlabels)
    
    tf.summary.image('example_labels_true', tf.concat([gt1, gt2, gt3], axis=0))
    tf.summary.image('example_labels_pred', tf.concat([pred1, pred2, pred3], axis=0))
    tf.summary.image('example_images', tf.concat([img1, img2, img3], axis=0))

    return supervised_loss, mean_dice

# ================================================================
# ================================================================
def evaluation_l2l(logits,
                   labels,
                   labels_masked,
                   nlabels,
                   loss_type,
                   are_labels_1hot = False):
    
    # =================
    # compute loss and foreground dice
    # =================
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits,
                                                                                         labels,
                                                                                         nlabels,
                                                                                         loss_type,
                                                                                         are_labels_1hot)


    mask_predicted = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)    
    
    if are_labels_1hot is False:
        mask_gt = labels
        mask_masked = labels_masked
    else:
        mask_gt = tf.argmax(labels, axis=-1)
        mask_masked = tf.argmax(labels_masked, axis=-1)

    # =================
    # write some segmentations to tensorboard
    # =================
    z=0
    z_idx = [10+z,20+z,30+z,40+z,50+z]
    gt1 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    gt4 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_z=z_idx[3], nlabels=nlabels)
    gt5 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_z=z_idx[4], nlabels=nlabels)
    
    gt1_masked = prepare_tensor_for_summary(mask_masked, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    gt2_masked = prepare_tensor_for_summary(mask_masked, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    gt3_masked = prepare_tensor_for_summary(mask_masked, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    gt4_masked = prepare_tensor_for_summary(mask_masked, mode='mask', n_idx_z=z_idx[3], nlabels=nlabels)
    gt5_masked = prepare_tensor_for_summary(mask_masked, mode='mask', n_idx_z=z_idx[4], nlabels=nlabels)
    
    pred1 = prepare_tensor_for_summary(mask_predicted, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(mask_predicted, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(mask_predicted, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    pred4 = prepare_tensor_for_summary(mask_predicted, mode='mask', n_idx_z=z_idx[3], nlabels=nlabels)
    pred5 = prepare_tensor_for_summary(mask_predicted, mode='mask', n_idx_z=z_idx[4], nlabels=nlabels)
    
    tf.summary.image('example_labels_true_masked_pred1', tf.concat([gt1, gt1_masked, pred1], axis=0))
    tf.summary.image('example_labels_true_masked_pred2', tf.concat([gt2, gt2_masked, pred2], axis=0))
    tf.summary.image('example_labels_true_masked_pred3', tf.concat([gt3, gt3_masked, pred3], axis=0))
    tf.summary.image('example_labels_true_masked_pred4', tf.concat([gt4, gt4_masked, pred4], axis=0))
    tf.summary.image('example_labels_true_masked_pred5', tf.concat([gt5, gt5_masked, pred5], axis=0))

    return supervised_loss, mean_dice

# ================================================================
# ================================================================
def evaluation_l2i(labels,
                   nlabels,
                   predicted_images,
                   true_images,
                   loss_type,
                   are_labels_1hot):


    if loss_type is 'l2':
        loss = tf.reduce_mean(tf.square(predicted_images - true_images))    
        
    elif loss_type is 'ssim':    
        loss = 1 - tf.reduce_mean(tf.image.ssim(img1 = predicted_images,
                                                img2 = true_images,
                                                max_val = 1.0))
    
    if are_labels_1hot is False:
        masks = labels
    else:
        masks = tf.argmax(labels, axis=-1)
    
    # =================
    # write some segmentations to tensorboard
    # =================
    mask1 = prepare_tensor_for_summary(masks, mode='mask', n_idx_batch=0, nlabels=nlabels)
    mask2 = prepare_tensor_for_summary(masks, mode='mask', n_idx_batch=1, nlabels=nlabels)
    mask3 = prepare_tensor_for_summary(masks, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    image_gt1 = prepare_tensor_for_summary(true_images, mode='image', n_idx_batch=0, nlabels=nlabels)
    image_gt2 = prepare_tensor_for_summary(true_images, mode='image', n_idx_batch=1, nlabels=nlabels)
    image_gt3 = prepare_tensor_for_summary(true_images, mode='image', n_idx_batch=2, nlabels=nlabels)
    
    image_pred1 = prepare_tensor_for_summary(predicted_images, mode='image', n_idx_batch=0, nlabels=nlabels)
    image_pred2 = prepare_tensor_for_summary(predicted_images, mode='image', n_idx_batch=1, nlabels=nlabels)
    image_pred3 = prepare_tensor_for_summary(predicted_images, mode='image', n_idx_batch=2, nlabels=nlabels)
    
    tf.summary.image('example_labels', tf.concat([mask1, mask2, mask3], axis=0))
    tf.summary.image('example_images_true', tf.concat([image_gt1, image_gt2, image_gt3], axis=0))
    tf.summary.image('example_images_pred', tf.concat([image_pred1, image_pred2, image_pred3], axis=0))

    return loss

# ================================================================
# ================================================================
def prepare_tensor_for_summary(img,
                               mode,
                               n_idx_batch=0,
                               n_idx_z=60,
                               nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, n_idx_z, 0, 0), (1, 1, -1, -1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else: raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8) # (1,224,224)
    V = tf.squeeze(V)
    V = tf.expand_dims(V, axis=0)
    
    # gather
    if mode == 'mask':
        cmap = 'viridis'
        cm = matplotlib.cm.get_cmap(cmap)
        colors = tf.constant(cm.colors, dtype=tf.float32)
        V = tf.gather(colors, tf.cast(V, dtype=tf.int32)) # (1,224,224,3)
        
    elif mode == 'image':
        V = tf.reshape(V, tf.stack((-1, tf.shape(img)[1], tf.shape(img)[2], 1))) # (1,224,224,1)
    
    return V
