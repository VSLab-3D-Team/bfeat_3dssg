import torch
import torch.nn as nn
import torch.nn.functional as F

class NegativeCosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity used in the SimSiam[0] paper.
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    Examples:
        >>> # initialize loss function
        >>> loss_fn = NegativeCosineSimilarity()
        >>>
        >>> # generate two representation tensors
        >>> # with batch size 10 and dimension 128
        >>> x0 = torch.randn(10, 128)
        >>> x1 = torch.randn(10, 128)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(x0, x1)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity
        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return -F.cosine_similarity(x0, x1, self.dim, self.eps).mean()


class SupervisedCrossModalInfoNCE(nn.Module):
    def __init__(self, device, temperature=0.07):
        super(SupervisedCrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True) # B X 1 
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        gt_label: torch.Tensor,
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # gt_label: B X C
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        labels = torch.argmax(gt_label, dim=1, keepdim=True)
        positive_mask = (labels == labels.T).float().to(self.device)
        negative_mask = (~(labels == labels.T)).float().to(self.device)
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            neg_sim = exp_sim_mat * valid_mask * negative_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat * negative_mask
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)
        

class CrossModalInfoNCE(nn.Module):
    """
    Get cross-modality wo/ self-contrastive settings.
    Calculate only 3D-Text Cross-Modality
    """
    def __init__(self, device, temperature=0.07):
        super(CrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True) # B X 1 
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        positive_mask = torch.eye(B).float().to(self.device)
        negative_mask = (~(positive_mask.bool())).float().to(self.device)
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            neg_sim = exp_sim_mat * valid_mask * negative_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat * negative_mask
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)

class CrossModalNXTent(nn.Module):
    """
    Get cross-modality wo/ self-contrastive settings.
    Calculate only 3D-Text Cross-Modality
    """
    def __init__(self, device, temperature=0.07):
        super(CrossModalInfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def __nonzero_mean(self, x: torch.Tensor):
        """
        Crazy Indexing for accurate mean
        Holy Shit
        """
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        count_non_zero = non_zero_mask.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) # B X 1 X 1
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def __text_nonzero_mean(self, x: torch.Tensor):
        non_zero_mask = x != 0
        sum_non_zero = x.sum(dim=1, keepdim=True) # B X 1
        count_non_zero = non_zero_mask.sum(dim=1, keepdim=True) # B X 1 
        mean_non_zero = torch.where(count_non_zero > 0, sum_non_zero / count_non_zero, torch.zeros_like(sum_non_zero).to(self.device))
        return mean_non_zero
    
    def forward(
        self, 
        z_p: torch.Tensor, 
        z_c: torch.Tensor,  
        mask: torch.Tensor = None
    ):
        # z_p: B X N_feat
        # z_c: 
        #   - B X K X N_feat if Image 
        #   - B X N_feat     if Text
        # mask: B X K \in {0, 1}
        B = z_p.shape[0]
        z_p = F.normalize(z_p, dim=-1)
        z_c = F.normalize(z_c, dim=-1)
        positive_mask = torch.eye(B).float().to(self.device)
        negative_mask = (~(positive_mask.bool())).float().to(self.device)
        
        # Intra-Modal negative similarity 
        pc_neg_sim = torch.mm('bn,mn->bm', z_p, z_p)
        exp_pc_neg_sim = torch.exp(pc_neg_sim) * negative_mask
        
        if not mask == None: # if zero-mask is given, cross-modal loss w. RGB Image
            K = z_c.shape[1]
            negative_mask = negative_mask.unsqueeze(2).repeat(1, 1, K)
            positive_mask = positive_mask.unsqueeze(2).repeat(1, 1, K)
            valid_mask = mask.unsqueeze(1).repeat(1, B, 1)
            c_sim = torch.einsum('bn,mkn->bmk', z_p, z_c) / self.temperature
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            # Masking valid RGB Image which exists
            exp_pc_neg_sim = exp_pc_neg_sim.unsqueeze(2).repeat(1, 1, K)
            neg_sim = exp_sim_mat * valid_mask + exp_pc_neg_sim * valid_mask # B X B X K, non-zero w. negative sample
            l_neg_term = neg_sim.sum(1).sum(1) # B
            l_bp = exp_sim_mat / l_neg_term.unsqueeze(1).unsqueeze(1).repeat(1, B, K) # B X B X K
            l_bp = -torch.log(l_bp) * positive_mask * valid_mask
            loss_br = self.__nonzero_mean(l_bp)
            return torch.mean(loss_br)

        else: # otherwise, cross-modal loss w. Text
            c_sim = torch.mm(z_p, z_c.T) / self.temperature # B X B
            exp_sim_mat = torch.exp(c_sim)  # B X B X K
            
            neg_sim = exp_sim_mat + exp_pc_neg_sim
            l_neg_term = neg_sim.sum(1, keepdim=True) # B
            l_bp = exp_sim_mat / l_neg_term.repeat(1, B)
            l_bp = -torch.log(l_bp) * positive_mask
            loss_bt = self.__text_nonzero_mean(l_bp)
            return torch.mean(loss_bt)

class IntraModalBarlowTwinLoss(nn.Module):
    def __init__(self, _lambda=5e-3):
        super(IntraModalBarlowTwinLoss, self).__init__()
        self._lambda = _lambda
        
    def forward(
        self, 
        z_a: torch.Tensor, 
        z_b: torch.Tensor
    ):
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        
        z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        # 2. 크로스 상관행렬 C 계산 (DxD 크기, D는 임베딩 차원)
        batch_size = z_a.size(0)
        c = torch.mm(z_a.T, z_b) / batch_size  # 배치 크기로 나누어 평균 상관값 계산
        # 3. 손실 함수 항목 계산
        identity = torch.eye(c.size(0)).to(c.device)  # 단위 행렬 (대각선이 1인 행렬)
        # - 대각선 요소를 1에 가깝게 만드는 불변성(invariance) 항
        invariance_term = (c.diag() - 1).pow(2).sum()
        # - 대각선 외 요소들을 0에 가깝게 만드는 중복 감소(redundancy reduction) 항
        redundancy_term = ((c - identity).pow(2).sum() - invariance_term)
        # 4. 최종 손실 계산
        loss = invariance_term + self._lambda * redundancy_term
        return loss