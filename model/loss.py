import torch
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


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        NT-Xent Loss 계산
        Args:
            z_i: 첫 번째 인코딩된 벡터 (batch_size x feature_dim)
            z_j: 두 번째 인코딩된 벡터 (batch_size x feature_dim)
        Returns:
            NT-Xent Loss 값 (스칼라)
        """
        batch_size = z_i.size(0)    
        z = torch.cat([z_i, z_j], dim=0)  # 2 * batch_size x feature_dim
        print(z)
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # 2*batch_size x 2*batch_size
        sim_matrix /= self.temperature
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        print(sim_matrix)
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(z.device)
        print(labels.shape)
        loss = F.cross_entropy(sim_matrix, labels, reduction="mean")
        print(loss)
        return loss
