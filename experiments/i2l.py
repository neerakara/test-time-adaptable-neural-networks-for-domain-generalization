import model_zoo
import tensorflow as tf

# ======================================================================
# Model settings
# ======================================================================

# training dataset
train_dataset = 'HCPT1'
tr_str = 'tr' + train_dataset

# data aug settings
da_ratio = 0.25
sigma = 20
alpha = 500
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

# run number
run_number = 1
run_str = '_run' + str(run_number)

model_handle_i2l = model_zoo.unet2D_i2l
experiment_name_i2l = 'i2l_mapper/' + tr_str + da_str + run_str

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (224, 224)
nlabels = 15
loss_type_i2l = 'dice'

# ======================================================================
# training settings
# ======================================================================
max_epochs = 1000
batch_size = 16
learning_rate = 1e-3    
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 20
train_eval_frequency = 500
val_eval_frequency = 500
save_frequency = 500
continue_run = False
debug = False

# ======================================================================
# test settings
# ======================================================================
## iteration number to be loaded after training the model (setting this to zero will load the model with the best validation dice score)
load_this_iter = 0
batch_size_test = 1
test_dataset = 'CALTECH' # 'HCPT1' or 'HCPT2' or 'CALTECH' or 'PFIZER'

if test_dataset is 'HCPT1':
    image_depth = 256
elif test_dataset is 'HCPT2':
    image_depth = 256
elif test_dataset is 'CALTECH':
    image_depth = 256
