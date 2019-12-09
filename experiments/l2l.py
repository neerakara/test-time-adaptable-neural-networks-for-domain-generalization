import model_zoo
import tensorflow as tf

# ======================================================================
# Model settings
# ======================================================================

# training dataset
train_dataset = 'HCPT1'
tr_str = 'tr' + train_dataset

# run number
run_number = 1
run_str = '_run' + str(run_number)

# data aug settings
da_ratio = 0.25
sigma = 20
alpha = 1000
trans_min = -10
trans_max = 10
rot_min = -10
rot_max = 10
scale_min = 0.9
scale_max = 1.1
da_str = '_da' + str(da_ratio)

res_str = '/res2.8_0.7_0.7_64_256_256' 
model_str = '/unet3D_n4_l2l_with_skip_connections_except_first_layer'
model_handle_l2l = model_zoo.unet3D_n4_l2l_with_skip_connections_except_first_layer

# mask settings
mask_type = 'squares_jigsaw' # zeros / random_labels / jigsaw
mask_radius = 10 # The mask will be a square with side length twice this number 
num_squares = 200
is_num_masks_fixed = False
is_size_masks_fixed = False
mask_str = '_mask_' + mask_type + '_maxlen' + str(2*mask_radius) + 'x' + str(num_squares)

experiment_name_l2l = 'l2l_mapper/' + tr_str + da_str + res_str + model_str + mask_str + run_str

loss_type_l2l = 'dice' # crossentropy / dice

# ======================================================================
# data settings
# ======================================================================
data_mode = '3D'
image_size = (64, 256, 256) #z-x-y
image_depth = 256 # crop out size for the volume before rescaling
target_resolution_brain = (0.7, 0.7, 2.8) # x-y-z
nlabels = 15

# ======================================================================
# training settings
# ======================================================================
max_steps = 50000
batch_size = 1
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 100
train_eval_frequency = 1000
val_eval_frequency = 1000
save_frequency = 1000
continue_run = False
debug = False