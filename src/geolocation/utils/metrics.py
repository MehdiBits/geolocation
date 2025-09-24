import torch

def kl_to_uniform(p):
    """
    Compute KL divergence between each distribution p and the uniform distribution.

    Args:
        p (torch.Tensor): shape (batch_size, distribution_support)
                         Each row should be a probability distribution (sums to 1).

    Returns:
        torch.Tensor: KL divergence for each row, shape (batch_size,)
    """
    _ , support = p.shape

    # small epsilon for numerical stability (avoid log(0))
    eps = 1e-12
    p = p.clamp(min=eps)

    # entropy of p: H(p) = - sum p log p
    entropy = -(p * p.log()).sum(dim=1)

    # KL(p || uniform) = log(support) - H(p)
    kl = torch.log(torch.tensor(support, dtype=p.dtype, device=p.device)) - entropy

    return kl.cpu().detach().numpy()


