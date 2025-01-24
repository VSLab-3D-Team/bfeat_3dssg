import os

DATA_PATH = "/media/michael/hdd3/SceneGraph/3DSSG"
PREPROCESS_PATH = "/media/michael/hdd3/SceneGraph/BFeat3DSG"
PROJECT_PATH = os.path.abspath(".")

dataset_config = {
  "root": f"{DATA_PATH}/3DSSG_subset",
  "selection": "",
  "use_data_augmentation": False,
  "ignore_scannet_rel": True,
  "is_v2": True,
  "_label_file": ["labels.instances.align.annotated.v2.ply", "inseg.ply", "cvvseg.ply"],
  "label_file": "labels.instances.align.annotated.v2.ply",
  "data_augmentation": False,
  "num_points": 512,
  "num_points_union": 256,
  "disable_support_rel": False,
  "with_bbox": False,
  "discard_some": False,
  "load_cache": False,
  "sample_in_runtime": True,
  "sample_num_nn": 2,
  "sample_num_seed": 4,
  "class_choice": [],
  "max_edges": -1,
  "drop_edge": 0.5,
  "drop_edge_eval": 0.0,
  "use_object_weight": False
}

model_config = {
  "N_LAYERS": 2,
  "USE_SPATIAL": True,
  "WITH_BN": False,
  "USE_GCN": True,
  "use_2d_feats": True,
  "USE_CONTEXT": True,
  "USE_GCN_EDGE": True,
  "USE_REL_LOSS": True,
  "OBJ_PRED_FROM_GCN": True,
  "_GCN_TYPE": ["TRIP", "EAN"],
  "GCN_TYPE": "EAN",
  "_ATTENTION" : ["fat"],
  "ATTENTION": "fat",
  "DROP_OUT_ATTEN": 0.5,
  "multi_rel_outputs": True,
  "feature_transform": False,
  "point_feature_size": 512,
  "edge_feature_size":256,
  "clip_feat_dim": 512, 
  "lambda_o": 0.1,
  "DIM_ATTEN": 256,
  "_WEIGHT_EDGE": ["BG", "DYNAMIC", "OCCU", "NONE"],
  "WEIGHT_EDGE": "DYNAMIC",
  "OBJ_EDGE": "NONE",
  "_GCN_AGGR": ["add","mean","max"],
  "GCN_AGGR": "max",
  "w_bg": 1.0,
  "NONE_RATIO": 1.0,
  "NUM_HEADS": 8,
  "use_pretrain":"",
  "use_descriptor": True,
  "obj_label_path": f"{DATA_PATH}/3DSSG_subset/classes.txt",
  "rel_label_path": f"{DATA_PATH}/3DSSG_subset/relations.txt",
  "adapter_path": f"/data/wangziqin/project/3DSSG_Repo/clip_adapter/checkpoint/origin_mean.pth"
}

config_system = {
  "_NAME": ["SGFN", "Mmgnet"],
  "MODEL_PATH": f"{PROJECT_PATH}/src/model/model.py",
  "NAME": "BetterFeat",
  "PATH": f"{PROJECT_PATH}/output",
  "multi_view_root": f"{DATA_PATH}",
  "VERBOSE": False,
  "DEVICE": "cuda",
  "SEED": 2025,
  "MAX_EPOCHES": 100,
  "LR": 0.0001,
  "W_DECAY": False,
  "AMSGRAD":False,
  "LR_SCHEDULE": "Cosine",
  "GPU": [0, 1],
  "SAVE_INTERVAL": 2000,
  "VALID_INTERVAL": 10,
  "LOG_INTERVAL": 100,
  "LOG_IMG_INTERVAL": 100,
  "WORKERS": 4,
  "Batch_Size": 16, 
  "update_2d": False,
  "EVAL": False,
  "_EDGE_BUILD_TYPE": ["FC", "KNN"],
  "EDGE_BUILD_TYPE": "KNN",
  "WEIGHTING": True,
  "MODEL": model_config,
  "dataset": dataset_config
}
