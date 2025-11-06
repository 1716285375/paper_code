# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : core_trpo.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : TRPO核心算法实现
TRPO（Trust Region Policy Optimization）算法的核心计算函数
包括信任区域更新、共轭梯度等
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def flat_grad(output: torch.Tensor, inputs: tuple, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
    """
    计算梯度并展平为一维向量
    
    Args:
        output: 输出张量
        inputs: 输入参数元组
        create_graph: 是否创建计算图
        retain_graph: 是否保留计算图
    
    Returns:
        展平后的梯度向量
    """
    grads = torch.autograd.grad(
        outputs=output,
        inputs=inputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    # 过滤None并展平
    flat_grads = []
    for grad in grads:
        if grad is not None:
            flat_grads.append(grad.view(-1))
    if flat_grads:
        return torch.cat(flat_grads)
    else:
        return torch.tensor([], dtype=output.dtype, device=output.device)


def flat_params(params) -> torch.Tensor:
    """
    将参数展平为一维向量
    
    Args:
        params: 参数列表或生成器
    
    Returns:
        展平后的参数向量
    """
    if isinstance(params, (list, tuple)):
        return torch.cat([p.view(-1) for p in params])
    else:
        # 假设是参数生成器
        return torch.cat([p.view(-1) for p in params])


def flat_hessian(hessian_vec: tuple) -> torch.Tensor:
    """
    将Hessian向量展平
    
    Args:
        hessian_vec: Hessian向量元组
    
    Returns:
        展平后的Hessian向量
    """
    # 确定设备（从第一个非None的梯度获取）
    device = None
    for h in hessian_vec:
        if h is not None:
            device = h.device
            break
    
    # 如果没有找到设备，使用默认设备
    if device is None:
        device = torch.device("cpu")
    
    # 展平所有梯度，确保都在同一设备上
    flat_grads = []
    for h in hessian_vec:
        if h is not None:
            flat_grads.append(h.view(-1))
        else:
            # 创建空的张量，确保在同一设备上
            flat_grads.append(torch.tensor([], dtype=torch.float32, device=device))
    
    if flat_grads:
        return torch.cat(flat_grads)
    else:
        return torch.tensor([], dtype=torch.float32, device=device)


class TrustRegionUpdater:
    """
    TRPO信任区域更新器
    
    实现共轭梯度法和线搜索，用于TRPO更新
    """
    
    def __init__(
        self,
        model: nn.Module,
        dist_class,
        train_batch: dict,
        advantages: torch.Tensor,
        kl_threshold: float = 0.01,
        max_line_search_steps: int = 15,
        accept_ratio: float = 0.1,
        back_ratio: float = 0.8,
        critic_lr: float = 5e-3,
        cg_damping: float = 0.1,
        cg_max_iters: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        初始化信任区域更新器
        
        Args:
            model: 策略模型
            dist_class: 动作分布类
            train_batch: 训练批次数据
            advantages: 优势估计
            kl_threshold: KL散度阈值
            max_line_search_steps: 最大线搜索步数
            accept_ratio: 接受比率
            back_ratio: 回退比率
            critic_lr: Critic学习率
            cg_damping: 共轭梯度阻尼系数
            cg_max_iters: 共轭梯度最大迭代次数
            device: 设备
        """
        self.model = model
        self.dist_class = dist_class
        self.train_batch = train_batch
        self.advantages = advantages
        self.kl_threshold = kl_threshold
        self.max_line_search_steps = max_line_search_steps
        self.accept_ratio = accept_ratio
        self.back_ratio = back_ratio
        self.critic_lr = critic_lr
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.device = device or next(model.parameters()).device
        
        self.stored_params = None
        self.initial_params = None
        
    def get_actor_parameters(self):
        """获取actor参数"""
        return list(self.model.parameters())
    
    def get_critic_parameters(self):
        """获取critic参数（如果存在）"""
        # 假设模型有critic相关参数
        if hasattr(self.model, 'critic_parameters'):
            return self.model.critic_parameters()
        elif hasattr(self.model, 'value_head'):
            return list(self.model.value_head.parameters())
        else:
            # 如果没有单独的critic，返回空列表
            return []
    
    def compute_policy_loss(self) -> torch.Tensor:
        """
        计算策略损失（使用当前参数）
        
        Returns:
            策略损失
        """
        obs = self.train_batch.get("obs", self.train_batch.get("observations"))
        actions = self.train_batch.get("actions", self.train_batch.get("action"))
        
        if isinstance(obs, (list, tuple)):
            obs = torch.stack(obs)
        if isinstance(actions, (list, tuple)):
            actions = torch.stack(actions)
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        
        # 获取当前策略分布
        # Agent对象使用_forward方法，返回(dist, value)
        if hasattr(self.model, '_forward'):
            dist, _ = self.model._forward(obs)
        elif hasattr(self.model, 'forward'):
            # 如果是PyTorch模型，直接调用forward
            result = self.model(obs)
            if isinstance(result, tuple):
                logits, _ = result
                dist = self.dist_class(logits=logits)
            else:
                dist = self.dist_class(logits=result)
        else:
            # 回退：假设是logits
            logits = self.model(obs) if callable(self.model) else obs
            dist = self.dist_class(logits=logits)
        old_logprobs = self.train_batch.get("logprobs", self.train_batch.get("old_logprobs"))
        
        if isinstance(old_logprobs, (list, tuple)):
            old_logprobs = torch.stack(old_logprobs)
        if not isinstance(old_logprobs, torch.Tensor):
            old_logprobs = torch.as_tensor(old_logprobs, dtype=torch.float32, device=self.device)
        
        new_logprobs = dist.log_prob(actions)
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 计算损失
        loss = (ratio * self.advantages).mean()
        return loss
    
    def compute_kl(self) -> torch.Tensor:
        """
        计算KL散度
        
        Returns:
            KL散度
        """
        obs = self.train_batch.get("obs", self.train_batch.get("observations"))
        
        if isinstance(obs, (list, tuple)):
            obs = torch.stack(obs)
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        # 当前策略
        # Agent对象使用_forward方法，返回(dist, value)
        if hasattr(self.model, '_forward'):
            curr_dist, _ = self.model._forward(obs)
        elif hasattr(self.model, 'forward'):
            # 如果是PyTorch模型，直接调用forward
            result = self.model(obs)
            if isinstance(result, tuple):
                logits, _ = result
                curr_dist = self.dist_class(logits=logits)
            else:
                curr_dist = self.dist_class(logits=result)
        else:
            # 回退：假设是logits
            logits = self.model(obs) if callable(self.model) else obs
            curr_dist = self.dist_class(logits=logits)
        
        # 旧策略（从action_dist_inputs恢复）
        old_dist_inputs = self.train_batch.get("action_dist_inputs", self.train_batch.get("old_dist_inputs"))
        if old_dist_inputs is None:
            # 如果没有旧分布输入，使用当前分布作为旧分布（这不是最优的，但可以工作）
            old_dist = curr_dist
        else:
            # 从保存的分布输入重建旧分布
            if isinstance(old_dist_inputs, (list, tuple)):
                old_dist_inputs = torch.stack(old_dist_inputs)
            if not isinstance(old_dist_inputs, torch.Tensor):
                old_dist_inputs = torch.as_tensor(old_dist_inputs, dtype=torch.float32, device=self.device)
            # 确保维度正确
            if old_dist_inputs.dim() == 1:
                old_dist_inputs = old_dist_inputs.unsqueeze(0)
            old_dist = self.dist_class(logits=old_dist_inputs)
        
        # 使用PyTorch的kl_divergence方法
        kl = torch.distributions.kl.kl_divergence(old_dist, curr_dist)
        return kl.mean()
    
    def fisher_vector_product(self, p: torch.Tensor) -> torch.Tensor:
        """
        计算Fisher信息矩阵与向量的乘积（F * p）
        
        Args:
            p: 向量
        
        Returns:
            F * p
        """
        p = p.detach()
        # 重新计算KL散度（每次调用都需要新的计算图）
        kl = self.compute_kl()
        actor_params = self.get_actor_parameters()
        
        # 计算KL散度的梯度（保留计算图以便计算Hessian）
        kl_grads_flat = flat_grad(kl, actor_params, create_graph=True, retain_graph=True)
        
        # 计算 Hessian-vector product: H * p
        kl_grad_p = (kl_grads_flat * p).sum()
        kl_hessian_p = torch.autograd.grad(
            kl_grad_p,
            actor_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        kl_hessian_p_flat = flat_hessian(kl_hessian_p)
        
        return kl_hessian_p_flat + self.cg_damping * p
    
    def conjugate_gradients(self, b: torch.Tensor, residual_tol: float = 1e-10) -> torch.Tensor:
        """
        共轭梯度法求解 (F + lambda * I) * x = b
        
        Args:
            b: 右端向量
            residual_tol: 残差容忍度
        
        Returns:
            解向量 x
        """
        x = torch.zeros_like(b, device=self.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(self.cg_max_iters):
            Avp = self.fisher_vector_product(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Avp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < residual_tol:
                break
            
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def store_params(self):
        """保存当前参数"""
        actor_params = self.get_actor_parameters()
        self.stored_params = flat_params(actor_params).clone()
        self.initial_params = flat_params(actor_params).clone()
    
    def restore_params(self):
        """恢复保存的参数"""
        if self.stored_params is not None:
            actor_params = self.get_actor_parameters()
            vector_to_parameters(self.stored_params, actor_params)
    
    def set_params(self, flat_params_vec: torch.Tensor):
        """设置参数"""
        actor_params = self.get_actor_parameters()
        vector_to_parameters(flat_params_vec, actor_params)
    
    def update_actor(self, policy_loss: torch.Tensor) -> bool:
        """
        更新actor参数（使用信任区域方法）
        
        Args:
            policy_loss: 策略损失
        
        Returns:
            是否成功更新
        """
        # 保存当前参数
        self.store_params()
        
        # 计算策略梯度
        actor_params = self.get_actor_parameters()
        # 直接使用flat_grad计算梯度（不需要先调用torch.autograd.grad）
        policy_grad = flat_grad(policy_loss, actor_params, retain_graph=False)
        
        # 使用共轭梯度法求解自然梯度方向
        step_dir = self.conjugate_gradients(-policy_grad)  # 注意负号（因为是最大化）
        
        # 计算步长
        fisher_norm = policy_grad.dot(step_dir)
        if fisher_norm < 0:
            # 如果方向不对，不更新
            return False
        
        scale = torch.sqrt(2 * self.kl_threshold / (fisher_norm + 1e-8))
        full_step = scale * step_dir
        
        # 线搜索
        current_params = flat_params(actor_params)
        current_loss = policy_loss.detach().cpu().item()
        expected_improve = policy_grad.dot(full_step).item()
        
        if expected_improve < 1e-7:
            return False
        
        updated = False
        fraction = 1.0
        
        for i in range(self.max_line_search_steps):
            new_params = current_params + fraction * full_step
            self.set_params(new_params)
            
            new_loss = self.compute_policy_loss().detach().cpu().item()
            loss_improve = new_loss - current_loss
            kl = self.compute_kl().detach().cpu().item()
            
            # 检查是否满足条件
            if kl < self.kl_threshold and (loss_improve / expected_improve) >= self.accept_ratio and loss_improve > 0:
                updated = True
                break
            else:
                expected_improve *= self.back_ratio
                fraction *= self.back_ratio
        
        if not updated:
            # 恢复参数
            self.restore_params()
        
        return updated
    
    def update_critic(self, critic_loss: torch.Tensor):
        """
        更新critic参数（使用简单梯度下降）
        
        Args:
            critic_loss: Critic损失
        """
        critic_params = self.get_critic_parameters()
        if len(critic_params) == 0:
            return
        
        critic_grad = torch.autograd.grad(
            critic_loss,
            critic_params,
            allow_unused=True,
        )
        critic_grad_flat = flat_grad(critic_loss, critic_params)
        
        # 简单梯度下降
        new_params = parameters_to_vector(critic_params) - self.critic_lr * critic_grad_flat
        vector_to_parameters(new_params, critic_params)
    
    def update(self, policy_loss: torch.Tensor, critic_loss: Optional[torch.Tensor] = None, update_critic: bool = True):
        """
        执行一次完整的更新
        
        Args:
            policy_loss: 策略损失
            critic_loss: Critic损失（可选）
            update_critic: 是否更新critic
        """
        self.update_actor(policy_loss)
        if update_critic and critic_loss is not None:
            self.update_critic(critic_loss)

