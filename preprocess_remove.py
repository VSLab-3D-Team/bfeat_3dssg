from config import dataset_config, config_system, PREPROCESS_PATH
import h5py
from glob import glob
from tqdm import tqdm
import numpy as np

def __remove_small(obj_h5_list):
    remove_candidate = []
    for idx, _p in tqdm(enumerate(obj_h5_list), total=len(obj_h5_list)):
        with h5py.File(_p, "r") as f:
            pts = np.array(f["obj_point"])
            n_p = pts.shape[0]
            if n_p < dataset_config["num_points_union"]: remove_candidate.append(idx)
    print(
        "# of outlier of small point cloud:", 
        len(remove_candidate), 
        "of total", 
        len(obj_h5_list)
    )
    obj_h5_list = [ x for i, x in enumerate(obj_h5_list) if not i in remove_candidate ]
    return obj_h5_list

if __name__ == "__main__":
    
    t_obj_h5_list = __remove_small(glob(f"{PREPROCESS_PATH}/*/*/*.h5"))
    v_obj_h5_list = __remove_small(glob(f"{PREPROCESS_PATH}_val/*/*/*.h5"))
    
    ## 저장 어떻게 할지 생각좀...
    print("## Accumulating training scans...")
    for _p in tqdm(t_obj_h5_list, total=len(t_obj_h5_list)):
        
        pass
    
    print("## Accumulating validation scans...")
    for _p in tqdm(v_obj_h5_list, total=len(v_obj_h5_list)):
        
        pass