import torch
import torch.nn as nn
from .various_divergence import VariousDivergence
import math

class RANK_CKA(VariousDivergence):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized
        # OT parameters (copied from ot.py)
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-9
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'  # or 'euclidean', 'cosine'

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        
        # Cross-entropy loss with ground-truth labels
        loss = self.compute_cross_entropy_loss(outputs.logits, output_data["labels"])[0]
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute ranking CKA loss with OT
        kd_loss, log = self.compute_ranking_cka_ot_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        print("rank_cka_loss:", kd_loss)
        print("ce_loss:", loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        accuracy = self.compute_accuracy(logits, output_data["labels"])
        log["accuracy"] = accuracy

        # Convert float values to tensors before logging
        device = loss.device
        log = {
            "loss": loss,
            "accuracy": torch.tensor(accuracy, device=device),
        }

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def compute_ranking_cka_ot_loss(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Extract embeddings for student and teacher
        stu_embed_tokens = self.get_embedding_layer(distiller.student_model, "student")
        tea_embed_tokens = self.get_embedding_layer(distiller.teacher_model, "teacher")

        stu_input_embeds = stu_embed_tokens(input_data["input_ids"]).detach()
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"]).detach()

        # Normalize teacher embeddings
        norm_tea_index_embeds = tea_input_embeds / tea_input_embeds.std()
        
        # Project student embeddings to query space
        stu_q_hiddens = distiller.projectors["query"](stu_input_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # Define the layers to process (you can modify this list as needed)
        student_layers_to_process = [3, 7, 11]  # Skip layer 0 (embeddings)
        
        # Find best matching layers
        best_matching_layers = self.find_best_matching_layers(
            student_layers_to_process, outputs, teacher_outputs, 
            stu_q_hiddens, tea_k_hiddens
        )
        
        # Compute OT loss for each matched pair
        total_ot_loss = 0
        num_pairs = 0
        device = outputs.hidden_states[0].device
        
        for k, l in best_matching_layers.items():
            if l != -1:                 
                # Compute OT loss between student layer k and aligned teacher layer l
                ot_loss = self.compute_ot_loss_for_layer_pair(
                    outputs.hidden_states[k], teacher_outputs.hidden_states[l],
                    input_data["attention_mask"], input_data["teacher_attention_mask"],
                    distiller
                )
                
                total_ot_loss += ot_loss
                num_pairs += 1
        
        # Convert logging values to tensors
        log["ranking_cka_ot_loss"] = total_ot_loss
        log["num_layer_pairs"] = torch.tensor(num_pairs, device=device)
        
        '''# Average OT loss across all pairs
        if num_pairs > 0:
            avg_ot_loss = total_ot_loss / num_pairs
        else:
            avg_ot_loss = torch.tensor(0.0, device=outputs.hidden_states[0].device, requires_grad=True)
        
        log["ranking_cka_ot_loss"] = avg_ot_loss.item()
        log["num_layer_pairs"] = num_pairs'''
        
        return total_ot_loss, log

    def get_embedding_layer(self, model, model_type):
        """Extract embedding layer from different model architectures"""
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings()
        elif hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            return model.bert.embeddings.word_embeddings
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        else:
            raise NotImplementedError(f"Unsupported {model_type} model architecture for embedding extraction")

    def compute_align_matrix_layer_k(self, k, l, outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens):
        """Compute aligned hidden states for layer k (student) and layer l (teacher)"""
        stu_hiddens = outputs.hidden_states[k]
        tea_hiddens = teacher_outputs.hidden_states[l]
        
        # Normalize teacher hidden states
        norm_teacher_hiddens = tea_hiddens / tea_hiddens.std()
        tea_v_hiddens = norm_teacher_hiddens.float()
        
        # Compute attention alignment
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(tea_hiddens.shape[-1])
        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(stu_hiddens)
        
        return t2s_hiddens

    def find_best_matching_layers(self, student_layers, outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens):
        """Find best matching teacher layer for each student layer using CKA"""
        best_matching_layers = {}
        cka_loss_fn = CKALoss(eps=1e-8)
        
        for k in student_layers:
            if k == 11:
                best_matching_layers[k] = 31
                break
            index_list = [3*k-2, 3*k-1, 3*k, 3*k+1, 3*k+2]
            best_loss = float('inf')
            best_index = -1
            
            for l in index_list:
                if l < 0 or l >= len(teacher_outputs.hidden_states):
                    continue
                
                # Compute aligned hidden states
                t2s_hiddens = self.compute_align_matrix_layer_k(
                    k, l, outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens
                )
                
                # Compute CKA loss
                cka_loss = cka_loss_fn(t2s_hiddens, outputs.hidden_states[k])
                loss = 1 - cka_loss  # Convert CKA similarity to loss
                
                if loss < best_loss:
                    best_loss = loss
                    best_index = l
            
            best_matching_layers[k] = best_index
        
        return best_matching_layers

    def compute_ot_loss_for_layer_pair(self, student_hidden, teacher_hidden, student_mask, teacher_mask, distiller):
        """Compute OT loss between student and teacher hidden states for a single layer pair"""
        batch_size = student_hidden.size(0)
        total_loss = torch.tensor(0.0, device=student_hidden.device)
        
        # Check if projector exists
        if not hasattr(distiller, 'projectors') or 't2s' not in distiller.projectors:
            raise AttributeError("Distiller missing 't2s' projector. Make sure projectors are properly initialized.")
            
        projector = distiller.projectors["t2s"]
        
        for b in range(batch_size):
            # Get sequences for current batch
            teacher_seq = teacher_hidden[b]  # Shape: (seq_len, hidden_dim)
            student_seq = student_hidden[b]  # Shape: (seq_len, hidden_dim)

            # Get masks for current batch (assuming same masks for both)
            mask = student_mask[b]  # (seq_len)
            
            # Prune sequences based on the mask
            teacher_seq = teacher_seq[mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            student_seq = student_seq[mask.bool()]  # Shape: (valid_seq_len, hidden_dim)
            
            # Project teacher sequence to student space
            projected_teacher_seq = projector(teacher_seq)
            
            # Ensure both tensors are in the same dtype
            dtype = student_seq.dtype
            projected_teacher_seq = projected_teacher_seq.to(dtype)
            
            # Compute cost matrix based on specified distance metric
            if self.ot_dist_type == 'euclidean':
                cost_matrix = self.pairwise_euclidean_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'cosine':
                cost_matrix = self.pairwise_cosine_distance(student_seq, projected_teacher_seq)
            elif self.ot_dist_type == 'attention':
                cost_matrix = self.pairwise_attention_distance(student_seq, projected_teacher_seq)
            else:
                raise ValueError(f"Unknown distance type: {self.ot_dist_type}")
            
            # Ensure cost matrix is in the right dtype
            cost_matrix = cost_matrix.to(dtype)
            
            # Compute OT plan and loss
            ot_loss, transport_plan = self.sinkhorn(cost_matrix)
            total_loss += ot_loss
        
        avg_loss = total_loss / batch_size
        return avg_loss

    # Distance computation methods (copied from ot.py)
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        """Computes pairwise cosine distance with numerical stability"""
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=a.dtype))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=b.dtype))
        sim_mt = torch.mm(a_norm.transpose(0, 1), b_norm)
        sim_mt = 1 - sim_mt
        return sim_mt

    def pairwise_attention_distance(self, x, y, eps=1e-8):
        d = x.shape[1]
        sim_mt = torch.mm(x.transpose(0, 1), y) / math.sqrt(d)
        attention_weights = torch.softmax(sim_mt, dim=1)
        dist_mt = 1.0 - attention_weights
        return dist_mt
    
    def sinkhorn(self, cost_matrix, num_iters=None):
        """Sinkhorn algorithm for computing optimal transport"""
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        dtype = cost_matrix.dtype
        device = cost_matrix.device
        
        # Initialize uniform marginals
        a = torch.ones(m, device=device, dtype=dtype) / m
        b = torch.ones(n, device=device, dtype=dtype) / n
        
        # Initialize transport plan
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        
        # Initialize u and v
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            u_prev = u.clone()
            
            # Compute v first
            v = b / (torch.matmul(K.t(), u) + self.epsilon)
            # Then compute u
            u = a / (torch.matmul(K, v) + self.epsilon)
            
            # Check convergence
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        
        # Compute transport plan
        diag_u = torch.diag(u)
        diag_v = torch.diag(v)
        transport_plan = torch.matmul(torch.matmul(diag_u, K), diag_v)
        
        # Compute OT loss
        ot_loss = torch.sum(transport_plan * cost_matrix)
        
        return ot_loss, transport_plan


class CKALoss(nn.Module):
    """CKA Loss for measuring similarity between hidden representations"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        # Center the representations
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        # Compute CKA
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num / torch.sqrt(den1 * den2)
