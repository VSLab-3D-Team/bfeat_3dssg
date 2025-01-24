from model.frontend.pointnet import PointNetEncoder
from model.frontend.dgcnn import DGCNN
from dataset.database import SSGCMFeatDataset
from torch.utils.data import DataLoader
import argparse
import torch
from tqdm import tqdm
import pickle
import os

parser = argparse.ArgumentParser(description="Evaluation for feature space discriminative")
parser.add_argument("--exp_dir", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    
    model_path = args.exp_dir
    device = "cuda:1"
    encoder_model = PointNetEncoder(device, channel=9).to(device)
    encoder_model.load_state_dict(torch.load(model_path))
    
    v_dataset = SSGCMFeatDataset(split="validation_scans", use_rgb=True, use_normal=True, device=device)
    v_loader = DataLoader(v_dataset, batch_size=256, shuffle=True, drop_last=True)
    
    feat_per_labels = {}
    for i, (data_t1, data_t2, rgb_img, text_feat, label) in tqdm(enumerate(v_loader)):
        data_t1, data_t2, rgb_img, text_feat = \
                data_t1.to(device), data_t2.to(device), rgb_img.to(device).float(), text_feat.to(device)
        batch_size = data_t1.size()[0]
        
        data = torch.cat((data_t1, data_t2))
        data = data.transpose(2, 1).contiguous()
        point_feats, _, _ = encoder_model(data)
        
        point_t1_feats = point_feats[:batch_size, :]
        point_t2_feats = point_feats[batch_size:, :]
        z = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
        
        if not label in feat_per_labels.keys():
            feat_per_labels[label] = [z.cpu().numpy()]
        else:
            feat_per_labels[label].append(z.cpu().numpy())
    
    if not os.path.exists("./experiment/results"):
        os.makedirs("./experiment/results")
    
    with open("./experiment/results", "wb") as f:
        pickle.dump(feat_per_labels, f)