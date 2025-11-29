"""
Quality Loss for Stage 3
Pushes SNN outputs toward HQ prompt and away from LQ prompt
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLoss(nn.Module):
    """
    Prompt loss from SpikeCLIP paper (Eq. 9):
    
    L_prompt = -e^(Φimage(I)·Φtext(Thq)) / Σ(e^(Φimage(I)·Φtext(Ti)))
    
    where i ∈ {hq, lq}
    
    This maximizes the probability that the image aligns with HQ prompt.
    """

    def forward(self, image_features, text_features_hq, text_features_lq):
        """
        Compute prompt loss (Paper Eq. 9).

        :param image_features: (B, 512) normalized image features from CLIP
        :param text_features_hq: (512,) normalized HQ prompt features
        :param text_features_lq: (512,) normalized LQ prompt features
        :return: prompt loss (scalar)
        """
        # Compute similarities (no temperature scaling in Eq. 9)
        sim_hq = image_features @ text_features_hq  # (B,)
        sim_lq = image_features @ text_features_lq  # (B,)
        
        # Compute softmax probabilities: P(HQ | image)
        # exp(sim_hq) / (exp(sim_hq) + exp(sim_lq))
        exp_hq = torch.exp(sim_hq)
        exp_lq = torch.exp(sim_lq)
        prob_hq = exp_hq / (exp_hq + exp_lq)  # (B,)
        
        # Prompt loss: negative log probability of HQ
        # L_prompt = -log(P(HQ | image))
        loss = -torch.log(prob_hq + 1e-8).mean()
        
        return loss

class ClassLoss(nn.Module):
    """
    Class loss from SpikeCLIP paper (Eq. 10):
    
    L_class = -Σ log(e^(Φimage(Ii)·Φtext(Tci))/τ / Σ e^(Φimage(Ii)·Φtext(Tcj))/τ)
    
    InfoNCE loss for semantic alignment with class labels.
    """

    def __init__(self, temperature=0.07):
        """
        :param temperature: Temperature parameter τ for contrastive loss
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features_class):
        """
        Compute class loss (Paper Eq. 10).

        :param image_features: (B, 512) normalized image features from CLIP
        :param text_features_class: (B, 512) normalized class text features
        :return: class loss (scalar)
        """
        # Compute similarity matrix: (B, B)
        # image_features[i] should match text_features_class[i]
        logits = (image_features @ text_features_class.T) / self.temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(len(logits), device=logits.device)
        
        # InfoNCE loss: -log(exp(sim_pos) / Σ exp(sim_all))
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CombinedStage3Loss(nn.Module):
    """
    Combined loss for Stage 3 following SpikeCLIP paper (Eq. 11):
    
    L_total = L_class + λ * L_prompt
    
    where λ = 100 (as specified in paper)
    
    - L_class: InfoNCE loss for semantic alignment (Eq. 10)
    - L_prompt: Quality-guided prompt loss (Eq. 9)
    """

    def __init__(
            self,
            lambda_prompt=100.0,
            temperature=0.07
    ):
        super().__init__()
        self.lambda_prompt = lambda_prompt
        self.temperature = temperature

        self.prompt_loss = PromptLoss()
        self.class_loss = ClassLoss(temperature)

    def forward(
            self,
            image_features,
            text_features_hq,
            text_features_lq,
            text_features_class
    ):
        """
        Compute combined Stage 3 loss (Paper Eq. 11).

        :param image_features: (B, 512) CLIP features of SNN output
        :param text_features_hq: (512,) HQ prompt features
        :param text_features_lq: (512,) LQ prompt features
        :param text_features_class: (B, 512) Class text features (REQUIRED)
        :return: total loss, dict of individual losses
        """
        losses = {}

        # 1. Class Loss (InfoNCE with class labels) - Eq. 10
        loss_class = self.class_loss(image_features, text_features_class)
        losses['class'] = loss_class.item()

        # 2. Prompt Loss (Quality guidance) - Eq. 9
        loss_prompt = self.prompt_loss(image_features, text_features_hq, text_features_lq)
        losses['prompt'] = loss_prompt.item()

        # Combined loss (Paper Eq. 11): L_total = L_class + λ * L_prompt
        total_loss = loss_class + self.lambda_prompt * loss_prompt
        losses['total'] = total_loss.item()

        return total_loss, losses
