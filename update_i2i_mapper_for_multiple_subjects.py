import subprocess

for subject_id in range(20):
    subprocess.call(['python', '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/code/brain/v2.0/update_i2i_mapper.py', str(subject_id)])
