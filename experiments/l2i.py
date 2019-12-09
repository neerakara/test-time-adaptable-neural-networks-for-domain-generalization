import model_zoo
import tensorflow as tf

# ======================================================================
# Model settings
# ======================================================================
# train settings
train_dataset = 'HCPT1'
tr_str = 'tr' + train_dataset

# run number
run_number = 1
run_str = '_run' + str(run_number)

# training loss
loss_type_l2i = 'l2' # l2 / ssim
loss_str = '_loss' + loss_type_l2i

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

experiment_name = 'l2i_mapper/' + tr_str + loss_str + da_str + run_str
model_handle_l2i = model_zoo.unet2D_l2i

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth = 256
target_resolution_brain = (0.7, 0.7)
nlabels = 15

# ======================================================================
# training settings
# ======================================================================
max_steps = 20000
batch_size = 16
learning_rate = 1e-3    
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 100
train_eval_frequency = 1000
val_eval_frequency = 1000
save_frequency = 1000
continue_run = False
debug = False