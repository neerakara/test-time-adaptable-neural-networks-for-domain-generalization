import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## ======================================================================
## ======================================================================
def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], start_crop[3], 0),
                                     (-1, output_size[1], output_size[2], output_size[3], -1))
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], 0),
                                     (-1, output_size[1], output_size[2], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)

## ======================================================================
## ======================================================================    
def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)
       
## ======================================================================
# max pooling layer
## ======================================================================
def max_pool_layer2d(x,
                     kernel_size=2,
                     strides=2,
                     padding="SAME"):

    op = tf.layers.max_pooling2d(inputs=x,
                                 pool_size=kernel_size,
                                 strides=strides,
                                 padding=padding)
    
    return op
    
## ======================================================================
# conv layer with adaptive batch normalization: convolution, followed by batch norm, followed by activation
## ======================================================================
def conv2D_layer(x,
                 name,
                 kernel_size=3,
                 num_filters=32,
                 strides=1,
                 padding="SAME"):

    conv = tf.layers.conv2d(inputs=x,
                            filters=num_filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            name=name,
                            use_bias=False)    
    
    return conv

## ======================================================================
# conv layer with batch normalization: convolution, followed by batch norm, followed by activation
## ======================================================================
def conv2D_layer_bn(x,
                    name,
                    training,
                    kernel_size=3,
                    num_filters=32,
                    strides=1,
                    activation=tf.nn.relu,
                    padding="SAME"):

    conv = tf.layers.conv2d(inputs=x,
                            filters=num_filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            name=name,
                            use_bias=False)    
    
    conv_bn = tf.layers.batch_normalization(inputs=conv,
                                            name = name + '_bn',
                                            training = training)
    
    act = activation(conv_bn)

    return act

## ======================================================================
# deconv layer with adaptive batch normalization: convolution, followed by batch norm, followed by activation
## ======================================================================
def deconv2D_layer_bn(x,
                      name,
                      training,
                      kernel_size=3,
                      num_filters=32,
                      strides=2,
                      activation=tf.nn.relu,
                      padding="SAME"):

    conv = tf.layers.conv2d_transpose(inputs=x,
                                      filters=num_filters,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      name=name,
                                      strides=strides,
                                      use_bias=False)
    
    conv_bn = tf.layers.batch_normalization(inputs=conv,
                                            name=name + '_bn',
                                            training=training)
    
    act = activation(conv_bn)

    return act

## ======================================================================
# bilinear upsample
## ======================================================================
def bilinear_upsample2D(x,
                        size,
                        name):
    
    x_reshaped = tf.image.resize_bilinear(x,
                                          size,
                                          name=name)    
    
    return x_reshaped

## ======================================================================
# conv layer with batch normalization: convolution, followed by batch norm, followed by activation
## ======================================================================
def conv3D_layer_bn(x,
                    name,
                    training,
                    kernel_size=3,
                    num_filters=32,
                    activation=tf.nn.relu,
                    padding="SAME",
                    kernel_initializer=None):

    conv = tf.layers.conv3d(inputs=x,
                            filters=num_filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            name=name,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)
    
    conv_bn = tf.layers.batch_normalization(inputs=conv,
                                            name = name + '_bn',
                                            training = training)
    
    act = activation(conv_bn)

    return act

## ======================================================================
# max pooling layer
## ======================================================================
def max_pool_layer3d(x,
                     kernel_size=2,
                     strides=2,
                     padding="SAME"):

    op = tf.layers.max_pooling3d(inputs=x,
                                 pool_size=kernel_size,
                                 strides=strides,
                                 padding=padding)
    
    return op

## ======================================================================
# deconv layer 3d
## ======================================================================
def deconv3D_layer(x,
                   name,
                   num_filters=32,
                   kernel_size=4,
                   strides=2,
                   activation=tf.nn.relu,
                   padding="SAME"):

    deconv = tf.layers.conv3d_transpose(inputs=x,
                                        filters=num_filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        name=name,
                                        use_bias=False)

    return deconv

## ======================================================================
# conv layer without batch normalization or activation
## ======================================================================
def conv3D_layer(x,
                 name,
                 num_filters,
                 kernel_size=3,
                 padding="SAME"):

    conv = tf.layers.conv3d(inputs=x,
                            filters=num_filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            name=name,
                            use_bias=False)

    return conv

## ======================================================================
# bilinear upsampling in 3D
## ======================================================================
def bilinear_upsample3D(x,
                        name,
                        factor):

    # Taken from: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html

    with tf.variable_scope(name):

        b_size, x_size, y_size, z_size, c_size =  x.shape.as_list()

        x_size_new = x_size*factor
        y_size_new = y_size*factor
        z_size_new = z_size*factor

        # resize y-z
        squeeze_b_x = tf.reshape(x, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear( squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])

        # squeeze and 2d resize
        squeeze_b_z = tf.reshape( reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])

    return output_tensor

## ======================================================================
# bilinear upsampling in 3D
## ======================================================================
def bilinear_upsample3D_(x,
                         name,
                         factor_x,
                         factor_y,
                         factor_z):

    # Taken from: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html

    with tf.variable_scope(name):

        b_size, x_size, y_size, z_size, c_size =  x.shape.as_list()

        x_size_new = int(x_size*factor_x)
        y_size_new = int(y_size*factor_y)
        z_size_new = int(z_size*factor_z)

        # resize y-z
        squeeze_b_x = tf.reshape(x, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear( squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])

        # squeeze and 2d resize
        squeeze_b_z = tf.reshape( reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])

    return output_tensor