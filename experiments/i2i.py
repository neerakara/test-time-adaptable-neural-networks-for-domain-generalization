import model_zoo
import tensorflow as tf

# ======================================================================
# test settings
# ======================================================================
train_dataset = 'HCPT1' # STANFORD / CALTECH / HCPT2 / 'HCPT1'
test_dataset = 'CALTECH' # STANFORD / CALTECH / HCPT2
normalize = False

# ====================================================
# settings of the image to label mapper (a model for the task)
# ====================================================
model_handle_i2l = model_zoo.unet2D_i2l
tr_str = 'tr' + train_dataset
da_ratio = 0.25
da_str = '_da' + str(da_ratio)
run_str = '_run' + str(1)
expname_i2l = tr_str + da_str + run_str

# ====================================================
# settings of the label to image mapper (likelihood)
# ====================================================
model_handle_l2i = model_zoo.unet2D_l2i
tr_str = 'tr' + train_dataset
loss_str = '_loss' + 'l2'
da_str = '_da' + str(0.25)
run_str = '_run' + str(1)
expname_l2i = tr_str + loss_str + da_str + run_str

# ====================================================
# settings of the label to label mapper (prior)
# ====================================================
tr_str = 'tr' + train_dataset
da_str = '_da' + str(0.25)

downsampling_factor_x = 4
downsampling_factor_y = 1
downsampling_factor_z = 1

res_str = '/res2.8_0.7_0.7_64_256_256'

model_handle_l2l = model_zoo.unet3D_n4_l2l_with_skip_connections_except_first_layer
model_str = '/unet3D_n4_l2l_with_skip_connections_except_first_layer'

mask_type = 'squares_jigsaw'
mask_radius = 10 # The mask will be a square with side length twice this number 
num_squares = 200
mask_str = '_mask_' + mask_type + '_maxlen' + str(2*mask_radius) + 'x' + str(num_squares)
run_str = '_run' + str(1)
expname_l2l = tr_str + da_str + res_str + model_str + mask_str + run_str

# ====================================================
# normalizer architecture
# ====================================================
model_handle_normalizer = model_zoo.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
learning_rate = 1e-3
norm_batch_norm = False
arch_str = str(norm_num_hidden_layers) + '_' + str(norm_num_filters_per_layer) + '_k' + str(norm_kernel_size) + '_' + norm_activation + '_lr' + str(learning_rate) + '_bn' + str(int(norm_batch_norm)) + '/'

# ====================================================
# weights for the two losses
# ====================================================
loss_type_prior = 'dice'  # crossentropy/dice
lambda_prior = 1.0
num_prior_samples = 1
loss_type_likelihood = 'ssim' # l2/ssim
lambda_likelihood = 0.1
cae_atlas_ratio_threshold = 0.0
min_atlas_dice = 0.0
threshold_str = 'cae_atlas_ratio_' + str(cae_atlas_ratio_threshold) + 'min_atlas_' + str(min_atlas_dice)
loss_str = '_lossP_' + loss_type_prior + '_' + str(lambda_prior) + threshold_str + '_lossL_' + loss_type_likelihood + '_' + str(lambda_likelihood)

expname_normalizer = 'normalizer/test' + test_dataset 
expname_normalizer = expname_normalizer + '/i2l_' + expname_i2l + '/l2i_' + expname_l2i + '/l2l_' + expname_l2l
run_number = 1
run_str = '/run' + str(run_number)
expname_normalizer = expname_normalizer + '/norm_' + arch_str + loss_str + run_str

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth_hcp = 256
image_depth_caltech = 256
image_depth_stanford = 132
target_resolution_brain = (0.7, 0.7)
nlabels = 15
batch_size = 16

image_size_downsampled = (int(256 / downsampling_factor_x), int(256 / downsampling_factor_y), int(256 / downsampling_factor_z))
batch_size_downsampled = int(batch_size / downsampling_factor_x)

# ======================================================================
# training settings
# ======================================================================
continue_run = True
max_steps = int(image_depth_hcp / batch_size)*600 + 1
optimizer_handle = tf.train.AdamOptimizer
debug = False
check_ood_frequency = int(image_depth_hcp / batch_size)*25
vis_frequency = int(image_depth_hcp / batch_size)*75 # ensure that this is a multiple of "check_ood_frequency"