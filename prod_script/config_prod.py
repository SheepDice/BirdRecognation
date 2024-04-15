import warnings
import os
warnings.simplefilter("ignore")

def set_root_dir():
    root_path = os.getcwd().split('\\')
    while root_path[-1] != "reco_oiseau_jan24bds":
        root_path.pop()
    root_path = '\\'.join(root_path)
    os.chdir(root_path)