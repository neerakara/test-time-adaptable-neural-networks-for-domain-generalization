import tensorflow as tf
import numpy as np

## ======================================================================
## ======================================================================
def compute_dice(logits, labels, epsilon=1e-10):
    '''
    Computes the dice score between logits and labels
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :return: dice (per label, per image in the batch)
    '''

    with tf.name_scope('dice'):

        prediction = tf.nn.softmax(logits)
        intersection = tf.multiply(prediction, labels)
        
        reduction_axes = [1,2]        
        
        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes) 
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes) 
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)

        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)
        
        # =============================
        # if a certain label is missing in the GT of a certain image and also in the prediction,
        # dice[this_image,this_label] will be incorrectly computed as zero whereas it should be 1.
        # =============================
        
        # mean over all images in the batch and over all labels.
        mean_dice = tf.reduce_mean(dice)
        
        # mean over all images in the batch and over all foreground labels.
        mean_fg_dice = tf.reduce_mean(dice[:,1:])
        
    return dice, mean_dice, mean_fg_dice

## ======================================================================
## ======================================================================
def compute_dice_3d_without_batch_axis(prediction,
                                       labels,
                                       epsilon=1e-10):

    with tf.name_scope('dice_3d_without_batch_axis'):

        intersection = tf.multiply(prediction, labels)        
        
        reduction_axes = [0, 1, 2]                
        
        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes) 
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes) 
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)
        
        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)
        
        # mean over all images in the batch and over all labels.
        mean_fg_dice = tf.reduce_mean(dice[1:])
        
    return mean_fg_dice

## ======================================================================
## ======================================================================
def dice_loss(logits, labels):
    
    with tf.name_scope('dice_loss'):
        
        _, mean_dice, mean_fg_dice = compute_dice(logits, labels)
        
        # loss = 1 - mean_fg_dice
        loss = 1 - mean_dice

    return loss

## ======================================================================
## ======================================================================
def dice_loss_within_mask(logits, labels, mask):
    
    with tf.name_scope('dice_loss_within_mask'):
        
        _, mean_dice, mean_fg_dice = compute_dice(tf.math.multiply(logits, mask),
                                                  tf.math.multiply(labels, mask))
        
        # loss = 1 - mean_fg_dice
        loss = 1 - mean_dice

    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss(logits, labels):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    
    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels):

    labels_copy = np.copy(labels)
    
    # add a small number for log and normalize
    labels_copy = labels_copy + 1e-20
    labels_copy = labels_copy / tf.expand_dims(tf.reduce_sum(labels_copy, axis=-1), axis=-1)
    
    # compute cross-entropy 
    loss = - tf.reduce_mean(tf.reduce_sum(predicted_probabilities * tf.math.log(labels_copy), axis=-1))    
    
    return loss