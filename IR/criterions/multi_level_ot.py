import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoTokenizer
import re
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss


def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def normalize(value):
    """Safely normalize tensor values to prevent NaN issues"""
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Replace any existing NaN or Inf values
    value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Calculate mean and std with more numerical stability
    means = value.mean(dim=-1, keepdim=True)
    
    # Calculate std with epsilon for numerical stability
    var = torch.var(value, dim=-1, keepdim=True, unbiased=False) + epsilon
    stds = torch.sqrt(var)
    
    # Normalize with a minimum standard deviation threshold
    z_score_normalized = (value - means) / (stds + epsilon)
    
    # Clip values to prevent extreme outliers
    z_score_normalized = torch.clamp(z_score_normalized, min=-10.0, max=10.0)
    
    # Final check for NaN values
    return torch.nan_to_num(z_score_normalized, nan=0.0, posinf=10.0, neginf=-10.0)

def KL_wo(y_s, y_t, T=1):
    p_s = F.log_softmax(y_s/T, dim=-1)
    p_t = F.softmax(y_t/T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

def compute_embeddings(model, tokenizer, texts, args):
    """Compute embeddings for a list of texts using the given model and tokenizer"""
    # Ensure model is in evaluation mode for inference
    model.eval()
    
    # Get the device from the model
    device = next(model.parameters()).device
    
    
    if args.peft:  # LLM2Vec style model
        # Tokenize and ensure all tensors are on the correct device
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length)
        
        # Move all input tensors to the model's device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        token_embeddings = outputs.last_hidden_state           # [B, T, D]
        attention_mask = inputs['attention_mask']              # [B, T]
        
        # Expand mask: [B, T] → [B, T, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        
        # Masked sum then divide by actual token count
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)        # [B, D]
        sum_mask = input_mask_expanded.sum(dim=1)                                         # [B, 1]
        
        # Add epsilon to prevent division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-8)
        embeddings = sum_embeddings / sum_mask                                           # [B, D]
        
    else:  # BERT style model
        # Tokenize and ensure all tensors are on the correct device
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length)
        
        # Move all input tensors to the model's device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        
        
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize

    return embeddings

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2   
        
    def sinkhorn_normalized(self, x, n_iters=20):
        """Apply Sinkhorn normalization with safety checks for NaN values"""
        # Replace any NaN or Inf values
        x = torch.nan_to_num(x, nan=1e-8, posinf=1e6, neginf=-1e6)
        
        # Add small epsilon to all elements to avoid zeros
        x = x + 1e-8
        
        # Sinkhorn iterations with safety checks
        for _ in range(n_iters):
            # Row normalization
            row_sums = torch.sum(x, dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-8)  # Prevent division by zero
            x = x / row_sums
            
            # Check for NaN and fix
            x = torch.nan_to_num(x, nan=1e-8)
            
            # Column normalization
            col_sums = torch.sum(x, dim=0, keepdim=True)
            col_sums = torch.clamp(col_sums, min=1e-8)  # Prevent division by zero
            x = x / col_sums
            
            # Check for NaN and fix
            x = torch.nan_to_num(x, nan=1e-8)
            
        return x

    def manual_cdist(self, x, y, p=1):
        """Manual implementation of cdist that supports BFloat16 tensors"""
        n = x.size(0)
        m = y.size(0)
        
        # Convert inputs to float32 for computation
        x_float = x.to(torch.float32)
        y_float = y.to(torch.float32)
        
        # Compute pairwise distances more efficiently using broadcasting
        if p == 1:
            # L1 distance using broadcasting
            x_expanded = x_float.unsqueeze(1)  # [n, 1, d]
            y_expanded = y_float.unsqueeze(0)  # [1, m, d]
            result = torch.sum(torch.abs(x_expanded - y_expanded), dim=2)  # [n, m]
        else:
            # For other norms, fall back to loop
            result = torch.zeros(n, m, device=x.device, dtype=torch.float32)
            for i in range(n):
                for j in range(m):
                    result[i, j] = torch.sum(torch.abs(x_float[i] - y_float[j]))
        
        # Convert back to original dtype
        return result.to(x.dtype)

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=10):
        """Compute Sinkhorn loss with safety checks, handling BFloat16"""
        try:
            # Convert to float32 for processing if needed
            orig_dtype = x.dtype
            if orig_dtype == torch.bfloat16:
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                
            # Check inputs for NaN values
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                
            if torch.isnan(y).any() or torch.isinf(y).any():
                y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Compute distance matrix - use manual implementation for BFloat16
            if orig_dtype == torch.bfloat16:
                Wxy = self.manual_cdist(x, y, p=1)
            else:
                # Try to use torch.cdist
                try:
                    Wxy = torch.cdist(x, y, p=1)
                except RuntimeError as e:
                    # Fallback to manual implementation if cdist fails
                    print(f"Falling back to manual cdist implementation: {e}")
                    Wxy = self.manual_cdist(x, y, p=1)
            
            # Check distance matrix for extreme values
            if torch.isnan(Wxy).any() or torch.isinf(Wxy).any():
                print("Distance matrix contains NaN or Inf values")
                Wxy = torch.nan_to_num(Wxy, nan=0.0, posinf=1e6, neginf=-1e6)
                
            # Compute kernel with numerical stability
            # Clip distances to prevent exp overflow
            Wxy_clipped = torch.clamp(Wxy, max=10.0)
            K = torch.exp(-Wxy_clipped / max(epsilon, 1e-5))
            
            # Check kernel for NaN values
            if torch.isnan(K).any() or torch.isinf(K).any():
                print("Kernel matrix contains NaN or Inf values")
                K = torch.nan_to_num(K, nan=1e-8, posinf=1.0, neginf=0.0)
            
            # Apply Sinkhorn normalization
            P = self.sinkhorn_normalized(K, n_iters)
            
            # Compute loss with safety check
            loss = torch.sum(P * Wxy)
            
            # Final safety check
            if torch.isnan(loss) or torch.isinf(loss):
                print("Final Sinkhorn loss is NaN or Inf")
                return torch.tensor(0.0, device=x.device)
                
            # Convert back to original dtype
            return loss.to(orig_dtype)
            
        except Exception as e:
            print(f"Error in sinkhorn_loss: {e}")
            return torch.tensor(0.0, device=x.device)
            
    def forward(self, y_s, y_t):
        """Forward pass with safety handling"""
        try:
            # Check inputs - for embeddings, we expect 2D tensors [B, D]
            if y_s.dim() == 2 and y_t.dim() == 2:
                # Embeddings case - convert to distributions by applying softmax
                # Add a dummy dimension to make it [B, 1, D] for consistency
                y_s = y_s.unsqueeze(1)
                y_t = y_t.unsqueeze(1)
            
            if y_s.size(0) == 0 or y_t.size(0) == 0:
                return torch.tensor(0.0, device=y_s.device)
                
            softmax = nn.Softmax(dim=-1)
            
            # Apply softmax with temperature scaling
            p_s = softmax(y_s/self.T)
            p_t = softmax(y_t/self.T)
            
            # Check for NaN values after softmax
            p_s = torch.nan_to_num(p_s, nan=1e-8)
            p_t = torch.nan_to_num(p_t, nan=1e-8)
            
            # Print information about data type
            
            emd_loss = 0
            valid_samples = 0
            
            for i in range(p_s.shape[0]):
                try:
                    sample_loss = self.sinkhorn_loss(x=p_s[i], y=p_t[i])
                    
                    # Skip if loss is NaN or Inf
                    if torch.isnan(sample_loss) or torch.isinf(sample_loss):
                        continue
                        
                    emd_loss += 0.001 * sample_loss
                    valid_samples += 1
                except Exception as e:
                    print(f"Error processing sample {i} in Sinkhorn forward: {e}")
                    
            # Return mean loss if there are valid samples, otherwise return zero
            if valid_samples > 0:
                return emd_loss / valid_samples
            else:
                return torch.tensor(0.0, device=y_s.device)
                
        except Exception as e:
            print(f"Error in Sinkhorn forward pass: {e}")
            return torch.tensor(0.0, device=y_s.device)


class MULTI_LEVEL_OT(MultipleNegativesRankingLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.mnr_loss = MultipleNegativesRankingLoss(args)
    
    def forward(
        self, 
        distiller, 
        anchors,
        positives, 
        logging_output, 
        batch_denom, 
    ):
        """
        Forward pass for IR task using anchor-positive pairs
        Args:
            distiller: The distiller object containing student and teacher models
            anchors: List of query texts
            positives: List of positive document texts
            logging_output: Logging dictionary
            batch_denom: Batch denominator for logging
        """
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        student_tokenizer = distiller.student_tokenizer
        teacher_tokenizer = distiller.teacher_tokenizers
        
        log = {}
        
        # Ensure models are on the same device
        device = next(student_model.parameters()).device
        
        teacher_device = next(teacher_model.parameters()).device
        
        if device != teacher_device:

            teacher_model = teacher_model.to(device)
        
        # Compute base Multiple Negatives Ranking Loss
        base_loss, _ = self.mnr_loss.forward(
            distiller, anchors, positives, logging_output, batch_denom
        )
        
        # Compute student embeddings
        student_anchor_emb = compute_embeddings(student_model, student_tokenizer, anchors, self.args)
        student_pos_emb = compute_embeddings(student_model, student_tokenizer, positives, self.args)
        
        # Compute teacher embeddings (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_anchor_emb = compute_embeddings(teacher_model, teacher_tokenizer, anchors, self.args)
            teacher_pos_emb = compute_embeddings(teacher_model, teacher_tokenizer, positives, self.args)
        
        # Ensure all embeddings are on the same device
        target_device = student_anchor_emb.device
        student_pos_emb = student_pos_emb.to(target_device)
        teacher_anchor_emb = teacher_anchor_emb.to(target_device)
        teacher_pos_emb = teacher_pos_emb.to(target_device)
        
        
        # Compute distillation loss
        kd_loss, log = self.compute_multi_level_ot_distillation_loss(
            student_anchor_emb, student_pos_emb,
            teacher_anchor_emb, teacher_pos_emb,
            distiller, log
        )
        
        print("multi_level_ot_loss:", kd_loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * base_loss + self.kd_rate * kd_loss
        log["loss"] = loss
        log["base_mnr_loss"] = base_loss
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output
    
    def compute_multi_level_ot_distillation_loss(self, student_anchor_emb, student_pos_emb, 
                                                teacher_anchor_emb, teacher_pos_emb, 
                                                distiller, log):
        """
        Compute multi-level optimal transport distillation loss for embeddings
        """
        
        
        # Ensure all tensors are on the same device
        target_device = student_anchor_emb.device
        student_pos_emb = student_pos_emb.to(target_device)
        teacher_anchor_emb = teacher_anchor_emb.to(target_device)
        teacher_pos_emb = teacher_pos_emb.to(target_device)
        
        # Check for NaN or Inf in embeddings
        for name, emb in [("student_anchor", student_anchor_emb), ("student_pos", student_pos_emb),
                         ("teacher_anchor", teacher_anchor_emb), ("teacher_pos", teacher_pos_emb)]:
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"❌ {name} has NaN or Inf")
                
        # Replace NaN values with zeros before normalization
        student_anchor_emb = torch.nan_to_num(student_anchor_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        student_pos_emb = torch.nan_to_num(student_pos_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        teacher_anchor_emb = torch.nan_to_num(teacher_anchor_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        teacher_pos_emb = torch.nan_to_num(teacher_pos_emb, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize embeddings if they aren't already normalized
        if not self.args.peft:  # BERT embeddings are already normalized in compute_embeddings
            student_anchor_emb = F.normalize(student_anchor_emb, p=2, dim=1)
            student_pos_emb = F.normalize(student_pos_emb, p=2, dim=1)
            teacher_anchor_emb = F.normalize(teacher_anchor_emb, p=2, dim=1)
            teacher_pos_emb = F.normalize(teacher_pos_emb, p=2, dim=1)
        
        # Compute similarity matrices (this creates distributions over the batch)
        student_anchor_sim = torch.matmul(student_anchor_emb, student_anchor_emb.T)  # [B, B]
        student_pos_sim = torch.matmul(student_pos_emb, student_pos_emb.T)          # [B, B]
        teacher_anchor_sim = torch.matmul(teacher_anchor_emb, teacher_anchor_emb.T)  # [B, B]
        teacher_pos_sim = torch.matmul(teacher_pos_emb, teacher_pos_emb.T)          # [B, B]
        
        # Apply temperature scaling and softmax to create distributions
        temperature = 0.1
        student_anchor_dist = F.softmax(student_anchor_sim / temperature, dim=-1)
        student_pos_dist = F.softmax(student_pos_sim / temperature, dim=-1)
        teacher_anchor_dist = F.softmax(teacher_anchor_sim / temperature, dim=-1)
        teacher_pos_dist = F.softmax(teacher_pos_sim / temperature, dim=-1)
        
        # Calculate L1 loss between distributions
        anchor_l1_loss = torch.abs(student_anchor_dist - teacher_anchor_dist).mean()
        pos_l1_loss = torch.abs(student_pos_dist - teacher_pos_dist).mean()
        
        # Calculate KL divergence between distributions
        try:
            anchor_kl_loss = KL_wo(teacher_anchor_dist, student_anchor_dist) * 0.1
            pos_kl_loss = KL_wo(teacher_pos_dist, student_pos_dist) * 0.1
            
            if torch.isnan(anchor_kl_loss) or torch.isinf(anchor_kl_loss):
                anchor_kl_loss = torch.tensor(0.0, device=student_anchor_emb.device)
            if torch.isnan(pos_kl_loss) or torch.isinf(pos_kl_loss):
                pos_kl_loss = torch.tensor(0.0, device=student_anchor_emb.device)
                
        except Exception as e:
            print(f"Error computing KL divergence: {e}")
            anchor_kl_loss = torch.tensor(0.0, device=student_anchor_emb.device)
            pos_kl_loss = torch.tensor(0.0, device=student_anchor_emb.device)
        
        # Calculate Sinkhorn loss between distributions
        try:
            sinkhorn_loss_fn = Sinkhorn_seq()
            
            # Apply Sinkhorn loss to similarity distributions
            anchor_sinkhorn_loss = sinkhorn_loss_fn(teacher_anchor_dist, student_anchor_dist) * 0.1
            pos_sinkhorn_loss = sinkhorn_loss_fn(teacher_pos_dist, student_pos_dist) * 0.1
            
            if torch.isnan(anchor_sinkhorn_loss) or torch.isinf(anchor_sinkhorn_loss):
                anchor_sinkhorn_loss = torch.tensor(0.0, device=student_anchor_emb.device)
            if torch.isnan(pos_sinkhorn_loss) or torch.isinf(pos_sinkhorn_loss):
                pos_sinkhorn_loss = torch.tensor(0.0, device=student_anchor_emb.device)
                
        except Exception as e:
            print(f"Error computing Sinkhorn loss: {e}")
            anchor_sinkhorn_loss = torch.tensor(0.0, device=student_anchor_emb.device)
            pos_sinkhorn_loss = torch.tensor(0.0, device=student_anchor_emb.device)
        
        # Combine all losses
        l1_loss = (anchor_l1_loss + pos_l1_loss) / 2
        kl_loss = (anchor_kl_loss + pos_kl_loss) / 2
        sinkhorn_loss = (anchor_sinkhorn_loss + pos_sinkhorn_loss) / 2
        
        multi_level_ot_loss = l1_loss + kl_loss + sinkhorn_loss
        
        # Safety check for final loss
        if torch.isnan(multi_level_ot_loss) or torch.isinf(multi_level_ot_loss):
            print("Final loss is NaN or Inf, returning zero")
            multi_level_ot_loss = torch.tensor(0.0, device=student_anchor_emb.device)
        

        
        log["multi_level_ot_loss"] = multi_level_ot_loss
        log["l1_loss"] = l1_loss
        log["kl_loss"] = kl_loss
        log["sinkhorn_loss"] = sinkhorn_loss
        
        return multi_level_ot_loss, log
