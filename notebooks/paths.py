import os

data_root = os.environ['DATA_ROOT']
notebook_root = "%s/notebooks" % (data_root)
scripts_root = "%s/scripts" % (data_root)
images_root = "%s/images" % (data_root)
projects_root = "%s/projects" % (data_root)
results_root = "%s/results" % (data_root)
tmp_dir = "%s/tmp" % (data_root)

path = {}
path['notebook_root']=notebook_root
path['scripts_root']=scripts_root
path['images_root']=images_root
path['projects_root']=projects_root
path['results_root']=results_root
path['tmp_dir']=tmp_dir
