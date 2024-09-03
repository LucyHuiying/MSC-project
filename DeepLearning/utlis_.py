import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.long() 
            logprobs = F.log_softmax(x, dim=-1)

            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (x.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            return self.kl_div(logprobs, true_dist) / x.size(0)
        else:
            
            return F.cross_entropy(x, target)



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        features = F.normalize(features, p=2, dim=1)

        
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise ValueError("features contains nan or inf")
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            raise ValueError("labels contains nan or inf")

        similarities = torch.mm(features, features.t()) / self.temperature

        
        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            raise ValueError("similarities contains nan or inf")

        
        similarities = similarities.clamp(min=1e-9)

        exp_similarities = torch.exp(similarities)

        
        if torch.isnan(exp_similarities).any() or torch.isinf(exp_similarities).any():
            raise ValueError("exp_similarities contains nan or inf")

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        pos_similarities = exp_similarities.masked_select(~mask).view(labels.shape[0], -1)
        pos_similarities = pos_similarities.sum(dim=-1)

        
        if torch.isnan(pos_similarities).any() or torch.isinf(pos_similarities).any():
            raise ValueError("pos_similarities contains nan or inf")

        neg_similarities = exp_similarities.masked_select(mask).view(labels.shape[0], -1)
        neg_similarities = neg_similarities.sum(dim=-1)

        
        if torch.isnan(neg_similarities).any() or torch.isinf(neg_similarities).any():
            raise ValueError("neg_similarities contains nan or inf")

        denominator = pos_similarities + neg_similarities
        denominator = denominator.clamp(min=1e-9)

        losses = -torch.log(pos_similarities / denominator)

        
        if torch.isnan(losses).any() or torch.isinf(losses).any():
            raise ValueError("losses contains nan or inf")

        return losses.mean()


class DynamicWeightedCrossEntropy(nn.Module):
    def __init__(self, initial_weight=1.0, weight_increase_factor=1.1, max_weight=10.0):
        super(DynamicWeightedCrossEntropy, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.weight = initial_weight
        self.weight_increase_factor = weight_increase_factor
        self.max_weight = max_weight

    def forward(self, inputs, targets):
        loss = self.ce_loss(inputs, targets)
        # Apply dynamic weights based on target values
        weights = (targets * self.weight) + ((1 - targets) * 1.0)
        weighted_loss = loss * weights
        return weighted_loss.mean()

    def update_weight(self):
        self.weight = min(self.weight * self.weight_increase_factor, self.max_weight)