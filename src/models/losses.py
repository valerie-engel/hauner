import torch
import torch.nn as nn

class MaxMarginLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, logits_pos, logits_neg):
        loss = torch.clamp(self.margin - logits_pos + logits_neg, min=0.0)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    # def forward(self, x, target_idx, pos_idx, neg_idx):
    #     # but what about multiple pos./neg. pairs?
    #     logits_pos = self.similarity(x[target_idx], x[pos_idx])
    #     logits_neg = self.similarity(x[target_idx], x[pos_idx])
    def forward(self, x, y, mask_pos):
        logits = self.similarity(x, y)
        logits_pos = logits[mask_pos]
        logits_neg = logits[~mask_pos]
        
        print(f"mean logits for positive pairs: {logits_pos.mean(d)}\nmean logits for negative pairs: {logits_neg.mean()}")
        # cross entropy with multiple positives?