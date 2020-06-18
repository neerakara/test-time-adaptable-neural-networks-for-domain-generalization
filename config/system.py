import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# project dirs
# ==================================================================
project_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/'
bmic_data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/'
project_code_root = os.path.join(project_root, 'code/')
project_data_root = os.path.join(project_root, 'data/')

# ==================================================================
# data dirs
# ==================================================================
orig_data_root_hcp = os.path.join(bmic_data_root,'HCP/3T_Structurals_Preprocessed/')
orig_data_root_abide = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/data/preproc_data/abide/'
orig_data_root_pfizer = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/PfizerData/'
orig_data_root_ixi = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/data/preproc_data/ixi/'

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_hcp = os.path.join(project_data_root,'preproc_data/hcp/')
preproc_folder_abide = os.path.join(project_data_root,'preproc_data/abide/')
preproc_folder_pfizer = os.path.join(project_data_root,'preproc_data/pfizer/')
preproc_folder_ixi = os.path.join(project_data_root,'preproc_data/ixi/')

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'brain/v2.0/logdir/')
