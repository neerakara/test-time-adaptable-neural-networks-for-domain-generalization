import nibabel as nib
import numpy as np
import os
import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# ===================================================
# ===================================================
def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

# ===================================================
# ===================================================
def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

# ===================================================
# ===================================================
def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

# ===================================================
# ===================================================
def save_nii(img_path, data, affine, header=None):
    '''
    Shortcut to save a nifty file
    '''
    if header == None:
        nimg = nib.Nifti1Image(data, affine=affine)
    else:
        nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

# ===================================================
# ===================================================
def normalise_image(image, norm_type = 'div_by_max'):
    '''
    make image zero mean and unit standard deviation
    '''
    if norm_type == 'zero_mean':
        img_o = np.float32(image.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        normalized_img = np.divide((img_o - m), s)
        
    elif norm_type == 'div_by_max':
        perc1 = np.percentile(image,1)
        perc99 = np.percentile(image,99)
        normalized_img = np.divide((image - perc1), (perc99 - perc1))
        normalized_img[normalized_img < 0] = 0.0
        normalized_img[normalized_img > 1] = 1.0
    
    return normalized_img
    
# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes(seg_mask):
    
    seg_mask_modified = group_segmentation_classes_15(seg_mask)
    return seg_mask_modified

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes_15(a):
    """
    Args:
    label_data : Freesurfer generated Labels Data of a 3D MRI scan.
    Returns:
    relabelled_data
    """
    
    background_ids = [0] # [background]
    csf_ids = [24] # [csf]
    brainstem_ids = [16] # [brain stem]    
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    cerebral_cortex_ids = [3,42]
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72] # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85] # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm
    
    a = np.array(a, dtype = 'uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype = 'uint16')

    unique_ids = np.unique(a)    
    # print("Unique labels in the original segmentation mask:", unique_ids)
    
    for i in unique_ids:
        if (i in cerebral_gm_ids): b[a == i] = 3
        elif (i in cerebral_cortex_ids): b[a == i] = 3
        elif (i in accumbens_ids): b[a == i] = 3
        elif (i in background_ids): b[a == i] = 0
        elif (i in cerebellum_gm_ids): b[a == i] = 1
        elif (i in cerebellum_wm_ids): b[a == i] = 2
        elif (i in cerebral_wm_ids): b[a == i] = 4
        elif (i in misc_ids): b[a == i] = 4
        elif (i in thalamus_ids): b[a == i] = 5
        elif (i in hippocampus_ids): b[a == i] = 6
        elif (i in amygdala_ids): b[a == i] = 7
        elif (i in ventricle_ids): b[a == i] = 8    
        elif (i in choroid_plexus_ids): b[a == i] = 8    
        elif (i in caudate_ids): b[a == i] = 9
        elif (i in putamen_ids): b[a == i] = 10
        elif (i in pallidum_ids): b[a == i] = 11
        elif (i in ventral_DC_ids): b[a == i] = 12
        elif (i in csf_ids): b[a == i] = 13
        elif (i in brainstem_ids): b[a == i] = 14
        else:
            print('unknown id:', i)
            print('num_voxels:', np.shape(np.where(a==i))[1])
        
    print("Unique labels in the modified segmentation mask: ", np.unique(b))
    
    return b
    
# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_image_and_label(image, # 2d
                                      label,
                                      sigma,
                                      alpha,
                                      random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    
    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_image, distored_label

# ==================================================================
# taken from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
# ==================================================================   
def elastic_transform_label(label, # 2d
                            sigma,
                            alpha,
                            random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = label.shape
    
    # random_state.rand(*shape) generate an array of image size with random uniform noise between 0 and 1
    # random_state.rand(*shape)*2 - 1 becomes an array of image size with random uniform noise between -1 and 1
    # applying the gaussian filter with a relatively large std deviation (~20) makes this a relatively smooth deformation field, but with very small deformation values (~1e-3)
    # multiplying it with alpha (500) scales this up to a reasonable deformation (max-min:+-10 pixels)
    # multiplying it with alpha (1000) scales this up to a reasonable deformation (max-min:+-25 pixels)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
    
    return distored_label

# ===============================================================
# ===============================================================
def make_onehot(a):
    # taken from https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy/36960495
    ncols = a.max()+1
    out = np.zeros((a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out