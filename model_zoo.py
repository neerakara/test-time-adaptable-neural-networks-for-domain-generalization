# ======================================================================
# import stuff
# ======================================================================
import tensorflow as tf
from tfwrapper import layers

# ======================================================================
# 2D Unet for mapping from images to segmentation labels
# ======================================================================
def unet2D_i2l(images, nlabels, training_pl): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('i2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2], axis=-1)        
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
        conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
        conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

    return pred

# ======================================================================
# 2D Unet for mapping from labels to images
# ======================================================================
def unet2D_l2i(labels, training_pl): # labels should be provided as 1-hot

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('l2i_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=labels, name='conv1_1', num_filters=n1, training = training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2], axis=-1)
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
        conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')        
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
        conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=1, kernel_size=1)

    return pred

# ======================================================================
# 2D Unet for label autoencoder
# ======================================================================
def unet2D_l2l(inputs, nlabels, training_pl): 

    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer2d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv2D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv4_2)[1], tf.shape(conv4_2)[2]), name='upsample1')
        conv6_1 = layers.conv2D_layer_bn(x=upsample1, name='conv6_1', num_filters=n4, training=training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n4, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv3_2)[1], tf.shape(conv3_2)[2]), name='upsample2')    
        conv7_1 = layers.conv2D_layer_bn(x=upsample2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample2D(conv7_2, size = (tf.shape(conv2_2)[1], tf.shape(conv2_2)[2]), name='upsample3')    
        conv8_1 = layers.conv2D_layer_bn(x=upsample3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv2D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample2D(conv8_2, size = (tf.shape(conv1_2)[1], tf.shape(conv1_2)[2]), name='upsample4')  
        conv9_1 = layers.conv2D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv2D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv2D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        # print('Shape of input: ' + str(inputs.shape))        
        # print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        # print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        # print('Shape after 3rd max pooling layer: ' + str(pool3.shape))        
        # print('Shape after 4th max pooling layer / bottleneck layer: ' + str(pool4.shape))            
        # print('Shape after 1st upsampling block: ' + str(upsample1.shape))            
        # print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        # print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        # print('Shape after 4rd upsampling block: ' + str(upsample4.shape)) 
        # print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def unet3D_n4_l2l_no_skip_connections(inputs,
                                      nlabels,
                                      training_pl): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv3D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv3D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample3D(conv4_2, factor = 2, name='upsample1')
        conv5_1 = layers.conv3D_layer_bn(x=upsample1, name='conv5_1', num_filters=n3, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample2')    
        conv6_1 = layers.conv3D_layer_bn(x=upsample2, name='conv6_1', num_filters=n2, training=training_pl)
        conv6_2 = layers.conv3D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv6_2, factor = 2, name='upsample3')   
        conv7_1 = layers.conv3D_layer_bn(x=upsample3, name='conv7_1', num_filters=n1, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv3D_layer(x=conv7_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        print('Shape of input: ' + str(inputs.shape))        
        print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        print('Shape after 3rd max pooling layer: ' + str(pool3.shape))        
        print('Shape of the bottleneck layer: ' + str(conv4_2.shape))            
        print('Shape after 1st upsampling block: ' + str(upsample1.shape))            
        print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def unet3D_n5_l2l_no_skip_connections(inputs,
                                      nlabels,
                                      training_pl): 

    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv3D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv3D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer3d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv3D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample1')
        conv6_1 = layers.conv3D_layer_bn(x=upsample1, name='conv6_1', num_filters=n3, training=training_pl)
        conv6_2 = layers.conv3D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv6_2, factor = 2, name='upsample2')
        conv7_1 = layers.conv3D_layer_bn(x=upsample2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv7_2, factor = 2, name='upsample3')    
        conv8_1 = layers.conv3D_layer_bn(x=upsample3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv3D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample3D(conv8_2, factor = 2, name='upsample4')   
        conv9_1 = layers.conv3D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv3D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        print('Shape of input: ' + str(inputs.shape))        
        print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        print('Shape after 3rd max pooling layer: ' + str(pool3.shape))        
        print('Shape after 4th max pooling layer: ' + str(pool4.shape))        
        print('Shape of the bottleneck layer: ' + str(conv5_2.shape))            
        print('Shape after 1st upsampling block: ' + str(upsample1.shape))            
        print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        print('Shape after 4th upsampling block: ' + str(upsample4.shape))     
        print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def unet3D_n5_l2l_with_skip_connections(inputs,
                                        nlabels,
                                        training_pl): 

    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv3D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv3D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer3d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv3D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample1')
        concat1 = tf.concat([upsample1, conv4_2], axis=-1)
        conv6_1 = layers.conv3D_layer_bn(x=concat1, name='conv6_1', num_filters=n3, training=training_pl)
        conv6_2 = layers.conv3D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv6_2, factor = 2, name='upsample2')
        concat2 = tf.concat([upsample2, conv3_2], axis=-1)
        conv7_1 = layers.conv3D_layer_bn(x=concat2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv7_2, factor = 2, name='upsample3')  
        concat3 = tf.concat([upsample3, conv2_2], axis=-1)
        conv8_1 = layers.conv3D_layer_bn(x=concat3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv3D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample3D(conv8_2, factor = 2, name='upsample4')   
        concat4 = tf.concat([upsample4, conv1_2], axis=-1)
        conv9_1 = layers.conv3D_layer_bn(x=concat4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv3D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        print('Shape of input: ' + str(inputs.shape))        
        print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        print('Shape after 3rd max pooling layer: ' + str(pool3.shape))        
        print('Shape after 4th max pooling layer: ' + str(pool4.shape))        
        print('Shape of the bottleneck layer: ' + str(conv5_2.shape))            
        print('Shape after 1st upsampling block: ' + str(upsample1.shape))            
        print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        print('Shape after 4th upsampling block: ' + str(upsample4.shape))     
        print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def unet3D_n5_l2l_with_skip_connections_except_first_layer(inputs,
                                                           nlabels,
                                                           training_pl): 

    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
        
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv3D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv3D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer3d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv3D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample1')
        concat1 = tf.concat([upsample1, conv4_2], axis=-1)
        conv6_1 = layers.conv3D_layer_bn(x=concat1, name='conv6_1', num_filters=n3, training=training_pl)
        conv6_2 = layers.conv3D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv6_2, factor = 2, name='upsample2')
        concat2 = tf.concat([upsample2, conv3_2], axis=-1)
        conv7_1 = layers.conv3D_layer_bn(x=concat2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv7_2, factor = 2, name='upsample3')  
        concat3 = tf.concat([upsample3, conv2_2], axis=-1)
        conv8_1 = layers.conv3D_layer_bn(x=concat3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv3D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample3D(conv8_2, factor = 2, name='upsample4')   
        conv9_1 = layers.conv3D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv3D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        print('Shape of input: ' + str(inputs.shape))        
        print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        print('Shape after 3rd max pooling layer: ' + str(pool3.shape))        
        print('Shape after 4th max pooling layer: ' + str(pool4.shape))        
        print('Shape of the bottleneck layer: ' + str(conv5_2.shape))            
        print('Shape after 1st upsampling block: ' + str(upsample1.shape))            
        print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        print('Shape after 4th upsampling block: ' + str(upsample4.shape))     
        print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def unet3D_n4_l2l_with_skip_connections_except_first_layer(inputs,
                                                           nlabels,
                                                           training_pl): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
                
        # ====================================
        # 4th Conv block
        # ====================================
        conv5_1 = layers.conv3D_layer_bn(x=pool3, name='conv5_1', num_filters=n4, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n4, training=training_pl)
                
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample2')
        concat2 = tf.concat([upsample2, conv3_2], axis=-1)
        conv7_1 = layers.conv3D_layer_bn(x=concat2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv7_2, factor = 2, name='upsample3')  
        concat3 = tf.concat([upsample3, conv2_2], axis=-1)
        conv8_1 = layers.conv3D_layer_bn(x=concat3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv3D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample3D(conv8_2, factor = 2, name='upsample4')   
        conv9_1 = layers.conv3D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv3D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        print('Shape of input: ' + str(inputs.shape))        
        print('Shape after 1st max pooling layer: ' + str(pool1.shape))
        print('Shape after 2nd max pooling layer: ' + str(pool2.shape))        
        print('Shape after 3rd max pooling layer: ' + str(pool3.shape))               
        print('Shape of the bottleneck layer: ' + str(conv5_2.shape))            
        print('Shape after 2nd upsampling block: ' + str(upsample2.shape))     
        print('Shape after 3rd upsampling block: ' + str(upsample3.shape))     
        print('Shape after 4th upsampling block: ' + str(upsample4.shape))     
        print('Shape of output (before softmax): ' + str(outputs.shape))

    return outputs

# ======================================================================
# normalization network
# ======================================================================
def net2D_i2i(images,
              exp_config,
              training):
        
    with tf.variable_scope('image_normalizer'):       
                
        num_layers = exp_config.norm_num_hidden_layers
        n1 = exp_config.norm_num_filters_per_layer
        k = exp_config.norm_kernel_size
        
        out = images
        
        for l in range(num_layers):
            out = tf.layers.conv2d(inputs=out,
                                   filters=n1,
                                   kernel_size=k,
                                   padding='SAME',
                                   name='norm_conv1_'+str(l+1),
                                   use_bias=True,
                                   activation=None)
            
            if exp_config.norm_batch_norm is True:
                out = tf.layers.batch_normalization(inputs=out, name = 'norm_conv1_' + str(l+1) + '_bn', training = training)
            
            if exp_config.norm_activation is 'elu':
                out = tf.nn.elu(out)
                
            elif exp_config.norm_activation is 'relu':
                out = tf.nn.relu(out)
                
            elif exp_config.norm_activation is 'rbf':            
                # ==================
                # fixed scale
                # ==================
                # scale = 0.2
                # ==================
                # learnable scale - one scale per layer
                # ==================
                # scale = tf.Variable(initial_value = 0.2, name = 'scale_'+str(l+1))
                # ==================
                # learnable scale - one scale activation unit
                # ==================
                init_value = tf.random_normal([1,1,1,n1], mean=0.2, stddev=0.05)
                scale = tf.Variable(initial_value = init_value, name = 'scale_'+str(l+1))
                out = tf.exp(-(out**2) / (scale**2))
        
        delta = tf.layers.conv2d(inputs=out,
                                  filters=1,
                                  kernel_size=k,
                                  padding='SAME',
                                  name='norm_conv1_'+str(num_layers+1),
                                  use_bias=True,
                                  activation=tf.identity)
        
        # =========================
        # Only model an additive residual effect with the normalizer
        # =========================
        output = images + delta
        
    return output, delta
