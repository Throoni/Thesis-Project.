"""
Custom Loss Functions for Financial Machine Learning

This module provides specialized loss functions for stock return prediction:
- HuberLoss: Robust to outliers in financial returns
- QuantileLoss: For distribution prediction
- AsymmetricLoss: Different penalties for over/under-prediction
- SharpeLoss: Directly optimize risk-adjusted returns
- ICLoss: Information Coefficient (rank correlation) loss

Author: Thesis Project
Date: December 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss) - robust to outliers.
    
    For financial returns with fat tails, Huber loss provides a balance
    between MSE (sensitive to outliers) and MAE (robust but less efficient).
    
    L(y, f(x)) = 
        0.5 * (y - f(x))^2           if |y - f(x)| <= delta
        delta * |y - f(x)| - 0.5 * delta^2   otherwise
    
    Args:
        delta: Threshold at which to switch from quadratic to linear loss
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(predictions, targets, beta=self.delta)


class QuantileLoss(nn.Module):
    """
    Quantile Loss for distribution prediction.
    
    Instead of predicting the mean, predict specific quantiles of the
    return distribution (e.g., median, 10th percentile, 90th percentile).
    
    L(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
    
    Args:
        quantile: The quantile to predict (0.5 = median, 0.1 = 10th percentile)
    """
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1"
        self.quantile = quantile
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = targets - predictions
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for different penalties on over/under-prediction.
    
    In finance, the cost of over-predicting returns (missing short opportunities)
    may differ from under-predicting (missing long opportunities).
    
    Args:
        alpha_over: Penalty weight for over-prediction (prediction > actual)
        alpha_under: Penalty weight for under-prediction (prediction < actual)
    """
    def __init__(self, alpha_over: float = 1.0, alpha_under: float = 1.5):
        super().__init__()
        self.alpha_over = alpha_over
        self.alpha_under = alpha_under
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = predictions - targets
        
        # Over-prediction: prediction > actual (errors > 0)
        over_pred_loss = self.alpha_over * F.relu(errors) ** 2
        
        # Under-prediction: prediction < actual (errors < 0)
        under_pred_loss = self.alpha_under * F.relu(-errors) ** 2
        
        return (over_pred_loss + under_pred_loss).mean()


class SharpeLoss(nn.Module):
    """
    Sharpe Ratio Loss - directly optimize risk-adjusted returns.
    
    Instead of minimizing prediction error, this loss maximizes the
    Sharpe ratio of a portfolio weighted by predictions.
    
    Sharpe = E[r_p] / std(r_p)
    
    We minimize negative Sharpe (to maximize Sharpe).
    
    Args:
        eps: Small constant for numerical stability
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # Normalize predictions to get portfolio weights
        weights = F.softmax(predictions.flatten(), dim=0)
        
        # Portfolio return (weighted average of actual returns)
        portfolio_return = (weights * returns.flatten()).sum()
        
        # For Sharpe, we need multiple periods - use batch as proxy
        # This is a simplified approximation
        weighted_returns = weights * returns.flatten()
        portfolio_std = weighted_returns.std() + self.eps
        
        # Negative Sharpe (we minimize this to maximize Sharpe)
        sharpe = portfolio_return / portfolio_std
        
        return -sharpe


class ICLoss(nn.Module):
    """
    Information Coefficient Loss - optimize rank correlation.
    
    IC (Spearman correlation) is the standard metric for evaluating
    cross-sectional return predictions. This loss directly optimizes IC.
    
    Uses a differentiable approximation of Spearman correlation.
    
    Args:
        eps: Small constant for numerical stability
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def _soft_rank(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Differentiable soft ranking using softmax."""
        # Pairwise differences
        diff = x.unsqueeze(-1) - x.unsqueeze(-2)  # [batch, n, n]
        
        # Soft indicator: probability that x_i > x_j
        soft_indicator = torch.sigmoid(diff / temperature)
        
        # Sum to get soft rank (1-indexed)
        soft_ranks = soft_indicator.sum(dim=-1) + 1
        
        return soft_ranks
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()
        
        # Get soft ranks
        pred_ranks = self._soft_rank(pred_flat)
        targ_ranks = self._soft_rank(targ_flat)
        
        # Center ranks
        pred_centered = pred_ranks - pred_ranks.mean()
        targ_centered = targ_ranks - targ_ranks.mean()
        
        # Pearson correlation on ranks (= Spearman correlation)
        numerator = (pred_centered * targ_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (targ_centered ** 2).sum()
        ) + self.eps
        
        ic = numerator / denominator
        
        # Return negative IC (minimize to maximize IC)
        return -ic


class CombinedLoss(nn.Module):
    """
    Combined Loss - weighted combination of multiple losses.
    
    Allows combining MSE with IC loss or other regularization terms.
    
    Args:
        losses: List of (loss_fn, weight) tuples
    """
    def __init__(self, losses: list):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for loss_fn, _ in losses])
        self.weights = [weight for _, weight in losses]
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(predictions, targets)
        return total_loss


class RankMSELoss(nn.Module):
    """
    Rank-weighted MSE Loss.
    
    Weights MSE by the rank of predictions - gives more weight to
    extreme predictions (top/bottom of ranking).
    
    Args:
        alpha: Power parameter for rank weighting
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()
        n = pred_flat.shape[0]
        
        # Get ranks of predictions (1 to n)
        _, indices = torch.sort(pred_flat)
        ranks = torch.zeros_like(pred_flat)
        ranks[indices] = torch.arange(1, n + 1, dtype=torch.float32, device=pred_flat.device)
        
        # Normalize ranks to [0, 1]
        norm_ranks = ranks / n
        
        # Weight: higher weight for extreme predictions
        weights = (torch.abs(norm_ranks - 0.5) * 2) ** self.alpha + 0.5
        weights = weights / weights.sum() * n  # Normalize to sum to n
        
        # Weighted MSE
        mse = (pred_flat - targ_flat) ** 2
        weighted_mse = (weights * mse).mean()
        
        return weighted_mse


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        name: Name of loss function ('mse', 'huber', 'quantile', 'asymmetric', 
              'sharpe', 'ic', 'rank_mse')
        **kwargs: Arguments to pass to loss function
    
    Returns:
        Loss function module
    """
    losses = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': HuberLoss,
        'quantile': QuantileLoss,
        'asymmetric': AsymmetricLoss,
        'sharpe': SharpeLoss,
        'ic': ICLoss,
        'rank_mse': RankMSELoss,
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(losses.keys())}")
    
    return losses[name.lower()](**kwargs)


# Convenience aliases
SmoothL1Loss = HuberLoss
PinballLoss = QuantileLoss

