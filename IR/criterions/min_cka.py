import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss
import math

class MIN_CKA(MultipleNegativesRankingLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

    def forward(
        self, 
        distiller, 
        anchors, 
        positives, 
        logging_output, 
        batch_denom, 
    ):
        """
        Compute MIN_CKA loss for IR tasks.
        - anchors: list of query strings
        - positives: list of positive document strings
        """
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        student_tokenizer = distiller.student_tokenizer
        
        # Get model's dtype and device
        dtype = next(student_model.parameters()).dtype
        device = next(student_model.parameters()).device
        
        # Encode anchors (queries) and positives (documents) with student model
        if self.args.peft:  # student is llm2vec (LLM-based model)
            # Process anchors (queries)
            anchor_inputs = student_tokenizer(
                anchors, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.args.max_length
            ).to(device)
            
            anchor_outputs = student_model(**anchor_inputs, output_hidden_states=True)
            
            # Mean pooling for anchors
            token_embeddings = anchor_outputs.last_hidden_state
            attention_mask = anchor_inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            emb_anchor = sum_embeddings / sum_mask
            
            # Process positives (documents)
            positive_inputs = student_tokenizer(
                positives, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.args.max_length
            ).to(device)
            
            positive_outputs = student_model(**positive_inputs, output_hidden_states=True)
            
            # Mean pooling for positives
            token_embeddings = positive_outputs.last_hidden_state
            attention_mask = positive_inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            emb_pos = sum_embeddings / sum_mask
            
        else:  # student is BERT
            # Process anchors (queries)
            anchor_inputs = student_tokenizer(
                anchors, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.args.max_length
            ).to(device)
            
            anchor_outputs = student_model(**anchor_inputs, output_hidden_states=True)
            emb_anchor = anchor_outputs.last_hidden_state[:, 0]  # [CLS] token
            
            # Process positives (documents)
            positive_inputs = student_tokenizer(
                positives, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.args.max_length
            ).to(device)
            
            positive_outputs = student_model(**positive_inputs, output_hidden_states=True)
            emb_pos = positive_outputs.last_hidden_state[:, 0]  # [CLS] token
            
            # Normalize embeddings for BERT
            emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
            emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        # Check for NaN or Inf values
        if torch.isnan(emb_anchor).any() or torch.isinf(emb_anchor).any():
            print("emb_anchor has NaN or Inf")
        if torch.isnan(emb_pos).any() or torch.isinf(emb_pos).any():
            print("emb_pos has NaN or Inf")

        # Compute Multiple Negatives Ranking Loss
        scores = torch.matmul(emb_anchor, emb_pos.T) * self.scale  # (B, B)
        labels = torch.arange(scores.size(0), device=device)
        ir_loss = F.cross_entropy(scores, labels)
        
        log = {}
        
        # Compute Knowledge Distillation loss using teacher model
        with torch.no_grad():
            teacher_model.eval()
            teacher_anchor_outputs = teacher_model(**anchor_inputs, output_hidden_states=True)
            teacher_positive_outputs = teacher_model(**positive_inputs, output_hidden_states=True)
        
        # Compute minimum CKA loss between student and teacher representations
        kd_loss, log = self.compute_min_cka_loss(
            anchor_outputs, positive_outputs, 
            teacher_anchor_outputs, teacher_positive_outputs,
            distiller, log
        )
        
        print("min_cka_loss:", kd_loss)
        print("ir_loss:", ir_loss)
        
        # Combine losses
        kd_loss = torch.tensor(kd_loss, device=device).to(dtype)
        total_loss = (1.0 - self.kd_rate) * ir_loss + self.kd_rate * kd_loss

        log["loss"] = total_loss

        # Update logging output
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return total_loss, {}
    
    def compute_min_cka_loss(
        self, 
        student_anchor_outputs, student_positive_outputs,
        teacher_anchor_outputs, teacher_positive_outputs, 
        distiller, log
    ):  
        device = next(distiller.student_model.parameters()).device
        
        # Combine anchor and positive outputs for processing
        # Process them together to maintain the dual-encoder structure
        student_outputs = self.combine_outputs(student_anchor_outputs, student_positive_outputs)
        teacher_outputs = self.combine_outputs(teacher_anchor_outputs, teacher_positive_outputs)
        
        # Project student embeddings to query space
        stu_q_hiddens = distiller.projectors["query"](student_outputs.hidden_states[-1]).float()
        tea_k_hiddens = teacher_outputs.hidden_states[-1].float() / teacher_outputs.hidden_states[-1].std()

        # Define the layers to process
        student_layers_to_process = [2, 7, 11] if len(student_outputs.hidden_states) > 11 else [1, 3, 5]
        
        # Find best matching layers and compute CKA loss directly
        total_cka_loss = 0
        num_pairs = 0
        cka_loss_fn = CKALoss(eps=1e-8)
        
        for k in student_layers_to_process:
            if k >= len(student_outputs.hidden_states):
                continue
                
            if k == max(student_layers_to_process):
                # Fixed mapping for the last layer
                best_teacher_layer = len(teacher_outputs.hidden_states) - 1
                # Compute CKA loss for fixed mapping
                t2s_hiddens = self.compute_align_matrix_layer_k(
                    k, best_teacher_layer, student_outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens
                )
                cka_similarity = cka_loss_fn(t2s_hiddens, student_outputs.hidden_states[k])
                pair_cka_loss = 1 - math.sqrt(cka_similarity)
                
                total_cka_loss += pair_cka_loss
                num_pairs += 1
            else:
                weight = []
                align_matrix = []
                # Find best matching teacher layer
                ratio = len(teacher_outputs.hidden_states) // len(student_outputs.hidden_states)
                center = k * ratio
                index_list = [max(0, center-2), max(0, center-1), center, 
                             min(len(teacher_outputs.hidden_states)-1, center+1), 
                             min(len(teacher_outputs.hidden_states)-1, center+2)]
                
                for l in index_list:
                    if l >= len(teacher_outputs.hidden_states):
                        continue
                    
                    # Compute aligned hidden states
                    t2s_hiddens = self.compute_align_matrix_layer_k(
                        k, l, student_outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens
                    )
                    align_matrix.append(t2s_hiddens)
                    
                    # Compute CKA loss (1 - CKA similarity)
                    cka_similarity = cka_loss_fn(t2s_hiddens, student_outputs.hidden_states[k])
                    weight.append(math.sqrt(cka_similarity))

                if weight:  # Check if we have valid weights
                    weight_norm = [w / sum(weight) for w in weight]
                    weighted_sum_matrix = sum(w * a for w, a in zip(weight_norm, align_matrix))
                    pair_cka_loss = 1 - math.sqrt(cka_loss_fn(weighted_sum_matrix, student_outputs.hidden_states[k]))
                    total_cka_loss += pair_cka_loss
                    num_pairs += 1
        
        # Convert logging values to tensors
        log["min_cka_loss"] = total_cka_loss
        log["num_layer_pairs"] = torch.tensor(num_pairs, device=device)
        
        return total_cka_loss, log

    def combine_outputs(self, anchor_outputs, positive_outputs):
        """Combine anchor and positive outputs for dual-encoder processing"""
        class CombinedOutputs:
            def __init__(self, anchor_outputs, positive_outputs):
                # Concatenate hidden states along batch dimension
                self.hidden_states = []
                for i in range(len(anchor_outputs.hidden_states)):
                    combined_hidden = torch.cat([
                        anchor_outputs.hidden_states[i],
                        positive_outputs.hidden_states[i]
                    ], dim=0)
                    self.hidden_states.append(combined_hidden)
                
                # Also combine last_hidden_state if available
                if hasattr(anchor_outputs, 'last_hidden_state') and hasattr(positive_outputs, 'last_hidden_state'):
                    self.last_hidden_state = torch.cat([
                        anchor_outputs.last_hidden_state,
                        positive_outputs.last_hidden_state
                    ], dim=0)
        
        return CombinedOutputs(anchor_outputs, positive_outputs)
    
    def compute_align_matrix_layer_k(self, k, l, student_outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens):
        """Compute aligned hidden states for layer k (student) and layer l (teacher)"""
        stu_hiddens = student_outputs.hidden_states[k]
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
        
        # Compute CKA similarity
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        # Return CKA similarity (not loss)
        return num / torch.sqrt(den1 * den2)
