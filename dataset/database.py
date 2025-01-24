import numpy as np
import torch
import torch.utils.data as data

from dataset.preprocess import rotation_matrix
from config import dataset_config, config_system, PREPROCESS_PATH
import clip
import h5py
from glob import glob

model, preprocess = clip.load("ViT-B/32", device='cuda')

class SSGCMFeatDataset(data.Dataset):
    def __init__(self, 
            split,
            use_rgb,
            use_normal,
            device
        ):
        super(SSGCMFeatDataset, self).__init__()
        assert split in ["train_scans", "validation_scans"], "Not the valid split"
        self.config = config_system
        self.mconfig = dataset_config
        self.root = self.mconfig["root"]
        self.split = split
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.obj_file_list = PREPROCESS_PATH if split == "train_scans" else PREPROCESS_PATH + "_val"
        self.device = device
        
        self.obj_h5_list = glob(f"{self.obj_file_list}/*/*/*.h5")
        
        # All activated on training
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
    
    def __data_augmentation(self, points):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = rotation_matrix([0, 0, 1], np.random.uniform(0, 2*np.pi, 1))
        centroid = points[:,:3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = np.dot(points[:, :3], matrix.T)
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset: 3 + ofset] = np.dot(points[:, ofset : 3 + ofset], matrix.T)
        return points
    
    def __len__(self):
        return len(self.obj_h5_list)
    
    def __to_torch(self, x):
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(self.device)
    
    def __read_compressed_file(self, _path: str):
        _data = {}
        with h5py.File(_path, "r") as f:
            _data["obj_point"] = self.__to_torch(f["obj_point"])
            _data["mv_rgb"] = []
            rgb_keys = [ x for x in list(f.keys()) if x.startswith("rgb_view") ]
            for k in rgb_keys:
                _data["mv_rgb"].append(self.__to_torch(f[k]))
            _data["instance_id"] = f.attrs["semantic_id"]
            _data["instance_name"] = f.attrs["semantic_name"]
        return _data
    
    
    def __getitem__(self, index):
        """
        Instance Point Cloud를 sampling하는 방법에 대한 조사가 필요함.
        Simsiam 방법론으로 어떻게 sampling 및 augmentation을 수행하지?
        
        - 25/01/23: First Experiment, Single Image view
        - ??/??/??: TODO: Multi-View pair settings
        """
        obj_path = self.obj_h5_list[index]
        obj_data = self.__read_compressed_file(obj_path)
        
        obj_pos_1 = self.__data_augmentation(obj_data["obj_point"])
        obj_pos_2 = self.__data_augmentation(obj_data["obj_point"])
        
        with torch.no_grad():
            image_feature = model.encode_image(obj_data["mv_rgb"][2]).to(self.device)
            text_feature = model.encode_text(clip.tokenize(f"A point cloud of a {obj_data['instance_name']}")).to(self.device)
        
        return obj_pos_1, obj_pos_2, image_feature, text_feature, obj_data["semantic_id"]