import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Standard InfoNCE (NT-Xent) contrastive loss for paired multimodal data.
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, ag_emb, ab_emb):
        """
        Inputs: 
        ag_emb: normalized Antigen embeddings (N, D)
        ab_emb: normalized Antibody embeddings (N, D)
        """
        # Compute cosine similarity matrix since vectors are L2-normalized
        logits = torch.matmul(ag_emb, ab_emb.T) / self.temperature
        
        N = ag_emb.size(0)
        # Ground truth matches are on the diagonal (i-th Ag matches i-th Ab in the paired batch)
        labels = torch.arange(N, device=ag_emb.device)
        
        # Contrastive loss across batch from both perspectives
        loss_ag = F.cross_entropy(logits, labels)
        loss_ab = F.cross_entropy(logits.T, labels)
        
        return (loss_ag + loss_ab) / 2
