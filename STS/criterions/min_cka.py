import torch
import torch.nn as nn
from .sts_loss import STSLoss
import math

class MIN_CKA(STSLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Ensure kd_rate is initialized

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
        
        # Get model's dtype
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True
        )
        
        predictions = outputs.scores
        log = {}
        loss_mse = nn.MSELoss()
        
        # STS task uses MSE loss with regression scores
        # Convert labels to model's dtype
        labels = output_data["labels"].to(dtype)
        loss = loss_mse(predictions, labels)
        
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute minimum CKA loss
        kd_loss, log = self.compute_min_cka_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        
        print("min_cka_loss:", kd_loss)
        print("mse_loss:", loss)
        
        # Combine losses
        kd_loss = torch.tensor(kd_loss, device=device).to(dtype)

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def compute_min_cka_loss(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):  
        student_dtype = next(distiller.student_model.parameters()).dtype
        teacher_dtype = next(distiller.teacher_model.parameters()).dtype
        device = next(distiller.student_model.parameters()).device
        # Project student embeddings to query space
        # Use the last hidden states from the outputs instead of input embeddings
        stu_q_hiddens = distiller.projectors["query"](outputs.hidden_states[-1]).float()
        tea_k_hiddens = teacher_outputs.hidden_states[-1].float() / teacher_outputs.hidden_states[-1].std()

        # Define the layers to process
        student_layers_to_process = [2,5,8,11]
        
        # Find best matching layers and compute CKA loss directly
        total_cka_loss = 0
        num_pairs = 0
        cka_loss_fn = CKALoss(eps=1e-8)
        
        for k in student_layers_to_process:
            if k == 11:
                # Fixed mapping for the last layer
                best_teacher_layer = 31
                # Compute CKA loss for fixed mapping
                t2s_hiddens = self.compute_align_matrix_layer_k(
                    k, best_teacher_layer, outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens
                )
                cka_similarity = cka_loss_fn(t2s_hiddens, outputs.hidden_states[k])
                pair_cka_loss = 1 - cka_similarity
                
                total_cka_loss += pair_cka_loss
                num_pairs += 1
            else:
                weight = []
                align_matrix = []
                # Find best matching teacher layer
                index_list = [3*k-2, 3*k-1, 3*k, 3*k+1, 3*k+2]
                # index_list = [3*k-1, 3*k, 3*k+1]
                # index_list = [3*k]
                best_cka_loss = float('inf')
                
                for l in index_list:
                    if l < 0 or l >= len(teacher_outputs.hidden_states):
                        continue
                    
                    # Compute aligned hidden states
                    t2s_hiddens = self.compute_align_matrix_layer_k(
                        k, l, outputs, teacher_outputs, stu_q_hiddens, tea_k_hiddens
                    )
                    align_matrix.append(t2s_hiddens)
                    
                    # Compute CKA loss (1 - CKA similarity)
                    cka_similarity = cka_loss_fn(t2s_hiddens, outputs.hidden_states[k])
                    weight.append(cka_similarity)

                weight_norm = [w / sum(weight) for w in weight]
                weighted_sum_matrix = sum(w * a for w, a in zip(weight_norm, align_matrix))
                pair_cka_loss = 1 - cka_loss_fn(weighted_sum_matrix, outputs.hidden_states[k])
                total_cka_loss += pair_cka_loss
                num_pairs += 1
        # Convert logging values to tensors
        log["min_cka_loss"] = total_cka_loss
        log["num_layer_pairs"] = torch.tensor(num_pairs, device=device)
        
        return total_cka_loss, log

    
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
