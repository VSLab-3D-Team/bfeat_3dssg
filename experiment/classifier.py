import torch
import torch.nn as nn
from experiment.dataset_cls import FeatPerLabelDataset
from torch.utils.data import TensorDataset, DataLoader
from experiment.eval_metrics import *
import pandas as pd

torch.manual_seed(2025)
# -------------------------------------------------
# 1. 하이퍼파라미터
# -------------------------------------------------
# Ours
input_dim   = 512          # feature 차원
# VL-SAT
# input_dim = 768
num_classes = 160
hidden_dims = (128, 256, 128)   # MLP 구조
batch_size  = 256
num_epochs  = 80
lr          = 1e-4
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
## Ours
# exp_name = "bfeat_mv_scl_closs_pointnet_2025-02-12_01_best_model"
# v_exp_name = "ours_full_validation"

## Ours wo/ Visual Modality
# exp_name = "bfeat_mv_scl_closs_pointnet_2025-02-12_01_best_model"
# v_exp_name = "ours_wo_visual"

## Ours wo/ Text Modality
# exp_name = "bfeat_wo_text_modal_pointnet_2025-05-05_23_ckpt_epoch_200"
# v_exp_name = "ours_wo_text"

## Ours wo/ Intra-Modality w/ Rotation Regularization
# exp_name = "bfeat_only_supervised_pointnet_2025-02-17_23_best_model"
# v_exp_name = "ours_wo_intra_w_reg"

## Ours wo/ Rotation Regularization
# exp_name = "bfeat_wo_rotation_reg_pointnet_2025-05-08_00_best_model"
# v_exp_name = "ours_wo_rot_reg"

## Ours w/ Vanilla SupCon loss
# exp_name = "nxtent_loss_posd_pointnet_2025-05-19_14_best_model"
# v_exp_name = "multi_infonce"

## Ours w/ InfoNCE loss + Class Numbers
# exp_name = "infonce_loss_pointnet_2025-05-19_19_best_model"
# v_exp_name = "vanilla_infonce"

## VL-SAT
# exp_name = "results_vlsat"
# v_exp_name = "vl_sat_validation"

exp_name = "fffinal_repro_pointnet_2025-05-20_00_best_model"
v_exp_name = "reproduce"

# -------------------------------------------------
# 2. 데이터 준비 (예시)
# -------------------------------------------------
t_dataset = FeatPerLabelDataset(
    pkl_dir=f"/media/michael/ssd1/SceneGraph/bfeat_object_experiments/{exp_name}/",        # feat_per_labels_000.pkl ...
    in_memory=True               # 데이터가 작으면 True 권장
)
train_loader = DataLoader(t_dataset, batch_size=512, shuffle=True, num_workers=4)

v_dataset = FeatPerLabelDataset(
    pkl_dir=f"./experiment/results/{v_exp_name}/",        # feat_per_labels_000.pkl ...
    in_memory=True               # 데이터가 작으면 True 권장
)
test_loader = DataLoader(v_dataset, batch_size=1, shuffle=True, num_workers=4)

# -------------------------------------------------
# 3. MLP 모델 정의
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, hidden_dims, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------------------------
# 4. 학습 루프
# -------------------------------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch:2d}/{num_epochs}, loss={loss.item():.4f}")


# -------------------------------------------------
# 6. 결과 출력
# -------------------------------------------------
topk, mean_acc = evaluate(model, test_loader)
print(f"\nTop-1 Accuracy : {topk[0]:.4f}")
print(f"Top-5 Accuracy : {topk[1]:.4f}")
print(f"Top-10 Accuracy: {topk[2]:.4f}")
print(f"Mean Class Acc : {mean_acc:.4f}")

overall, mean_cls, classwise_topk = evaluate_topk_mean_acc(model, test_loader, num_classes=160, topk=(1, 5, 10))

print("\n==== 결과 ====")
for k in (1, 5, 10):
    print(f"Top-{k:<2d}  Overall Acc = {overall[k]:.4f} | "
          f"Mean-Class Acc = {mean_cls[k]:.4f}")

df = pd.DataFrame({f"top{k}": acc.numpy() for k, acc in classwise_topk.items()})
df.index.name = "class_id"
df.to_csv(f"./experiment/results/{v_exp_name}/classwise_accuracy.csv")