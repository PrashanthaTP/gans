import torch
from torch import nn

class GanLoss(nn.Module):
    def __init__(self,real_label=1.0,fake_label=0.0,device='cpu'):
        super().__init__()
        self.register_buffer('real_label',torch.tensor(real_label))
        self.register_buffer('fake_label',torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device 
    def get_labels(self,preds,is_real_targets):
        labels = self.real_label if is_real_targets else self.fake_label
        return labels.expand_as(preds).to(self.device)
    
    def forward(self,preds,is_real_targets):
        labels = self.get_labels(preds,is_real_targets)
        return self.loss(preds,labels)