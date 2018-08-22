import os
from os.path import expanduser

class Paths:
    def __init__(self):
        self.data_root = os.environ['DATA_ROOT']
        
        self.notebook_root = "%s/notebooks" % (self.data_root)
        self.scripts_root = "%s/scripts" % (self.data_root)
        self.images_root = "%s/images" % (self.data_root)
        self.projects_root = "%s/projects" % (self.data_root)
        self.results_root = "%s/results" % (self.data_root)
        self.tmp_root = "%s/tmp" % (self.data_root)
        self.dat_root = "%s/dat" % (self.data_root)

        self.tmp_dir = '%s/tmp' % (expanduser("~"))
        self.ws_dir = '%s/workspace' % (expanduser("~"))

        self.mii_home = os.environ['MII_HOME']
        self.inhouse = '%s/src/inhouse' % (self.mii_home)

        self.path = {}
        self.path['data_root']=self.data_root
        self.path['notebook_root']=self.notebook_root
        self.path['scripts_root']=self.scripts_root
        self.path['images_root']=self.images_root
        self.path['projects_root']=self.projects_root
        self.path['results_root']=self.results_root
        self.path['tmp_root']=self.tmp_root
        self.path['dat_root']=self.dat_root
        self.path['tmp_dir']=self.tmp_dir
        self.path['ws_dir']=self.ws_dir
        self.path['mii_home']=self.mii_home
        self.path['inhouse']=self.inhouse

    def print_paths(self):
        print("Paths defined in paths module:")
        for k,v in self.path.items():
            print(" paths.%s: %s" % (k,v))
