import model_zoo
import tensorflow as tf

# training dataset
train_dataset = 'HCPT1' # CALTECH, STANFORD, HCPT1, HCPT2
tr_str = 'tr' + train_dataset

# run number
run_number = 3
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
gamma_min = 0.5
gamma_max = 2.0
brightness_min = 0.0
brightness_max = 0.1
noise_min = 0.0
noise_max = 0.1
da_str = '_da' + str(da_ratio)

# Model settings : i2i
model_handle_normalizer = model_zoo.net2D_i2i
norm_kernel_size = 1
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False
norm_arch_str = str(norm_num_hidden_layers) + '_' + str(norm_num_filters_per_layer) + '_k' + str(norm_kernel_size) + '_' + norm_activation + '_bn' + str(int(norm_batch_norm)) + '/'

# Model settings : i2l
model_handle_i2l = model_zoo.unet2D_i2l
experiment_name_i2l = 'i2l_mapper/' + norm_arch_str + tr_str + da_str + run_str

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth_hcp = 256
image_depth_caltech = 256
image_depth_ixi = 256
image_depth_stanford = 132
target_resolution_brain = (0.7, 0.7)
nlabels = 15
loss_type_i2l = 'dice'

# ======================================================================
# training settings
# ======================================================================
max_steps = 50001
batch_size = 16
learning_rate = 1e-3    
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 100
train_eval_frequency = 1000
val_eval_frequency = 1000
save_frequency = 1000
continue_run = False
debug = False
