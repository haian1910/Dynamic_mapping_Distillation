import torch
import torch.nn.functional as F
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss

class DualSpaceKDWithCMA(MultipleNegativesRankingLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate

    def forward(
        self, 
        distiller, 
        anchors, 
        positives, 
        logging_output, 
        batch_denom, 
    ):
        """
        Compute Dual-Space Knowledge Distillation with Cross-Model Attention for IR tasks.
        - anchors: list of queries
        - positives: list of positive documents
        """
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        student_tokenizer = distiller.student_tokenizer
        teacher_tokenizer = distiller.teacher_tokenizers
        
        # Ensure teacher model is on the same device as student model
        target_device = next(student_model.parameters()).device
        if next(teacher_model.parameters()).device != target_device:
            teacher_model = teacher_model.to(target_device)
        
        log = {}
        
        # Get student embeddings and representations
        student_emb_anchor, student_emb_pos, student_anchor_hiddens, student_pos_hiddens = self.get_student_representations(
            anchors, positives, student_model, student_tokenizer
        )
        
        # Compute base Multiple Negatives Ranking Loss
        base_loss = self.compute_mnr_loss(student_emb_anchor, student_emb_pos)
        
        # Get teacher embeddings and representations (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_emb_anchor, teacher_emb_pos, teacher_anchor_hiddens, teacher_pos_hiddens = self.get_teacher_representations(
                anchors, positives, teacher_model, teacher_tokenizer, target_device
            )
        
        # Compute dual-space KD loss with CMA
        kd_loss = self.compute_dual_space_kd_loss_with_cma(
            student_emb_anchor, student_emb_pos, student_anchor_hiddens, student_pos_hiddens,
            teacher_emb_anchor, teacher_emb_pos, teacher_anchor_hiddens, teacher_pos_hiddens,
            anchors, positives, distiller, log
        )
        
        print("dskd_cma_loss:", kd_loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * base_loss + self.kd_rate * kd_loss
        log["loss"] = loss
        log["base_loss"] = base_loss
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        
        return loss, logging_output

    def get_student_representations(self, anchors, positives, student_model, student_tokenizer):
        """Get embeddings and hidden states from student model"""
        device = next(student_model.parameters()).device
        
        if self.args.peft:  # student is llm2vec (SFT teacher)
            # Anchor representations
            anchor_inputs = student_tokenizer(
                anchors, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(device)
            
            anchor_outputs = student_model(**anchor_inputs, output_hidden_states=True)
            emb_anchor = self.mean_pooling(anchor_outputs.last_hidden_state, anchor_inputs['attention_mask'])
            anchor_hiddens = anchor_outputs.hidden_states[-1]  # Last layer hidden states
            
            # Positive representations
            pos_inputs = student_tokenizer(
                positives, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(device)
            
            pos_outputs = student_model(**pos_inputs, output_hidden_states=True)
            emb_pos = self.mean_pooling(pos_outputs.last_hidden_state, pos_inputs['attention_mask'])
            pos_hiddens = pos_outputs.hidden_states[-1]  # Last layer hidden states
            
        else:  # student is BERT
            # Anchor representations
            anchor_inputs = student_tokenizer(
                anchors, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(device)
            
            anchor_outputs = student_model(**anchor_inputs, output_hidden_states=True)
            emb_anchor = anchor_outputs.last_hidden_state[:, 0]  # [CLS] token
            anchor_hiddens = anchor_outputs.hidden_states[-1][:, 0, :]  # [CLS] hidden state
            
            # Positive representations
            pos_inputs = student_tokenizer(
                positives, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(device)
            
            pos_outputs = student_model(**pos_inputs, output_hidden_states=True)
            emb_pos = pos_outputs.last_hidden_state[:, 0]  # [CLS] token
            pos_hiddens = pos_outputs.hidden_states[-1][:, 0, :]  # [CLS] hidden state
            
            # Normalize embeddings
            emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
            emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        return emb_anchor, emb_pos, anchor_hiddens, pos_hiddens

    def get_teacher_representations(self, anchors, positives, teacher_model, teacher_tokenizer, target_device):
        """Get embeddings and hidden states from teacher model"""
        # Ensure teacher model is on target device
        teacher_model = teacher_model.to(target_device)
        
        # Anchor representations
        anchor_inputs = teacher_tokenizer(
            anchors, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(target_device)
        
        anchor_outputs = teacher_model(**anchor_inputs, output_hidden_states=True)
        
        # Determine teacher architecture and extract representations accordingly
        if hasattr(anchor_outputs, 'last_hidden_state') and len(anchor_outputs.last_hidden_state.shape) == 3:
            # LLM-based teacher - use mean pooling
            emb_anchor = self.mean_pooling(anchor_outputs.last_hidden_state, anchor_inputs['attention_mask'])
            anchor_hiddens = anchor_outputs.hidden_states[-1]
        else:
            # BERT-based teacher - use [CLS] token
            emb_anchor = anchor_outputs.last_hidden_state[:, 0]
            anchor_hiddens = anchor_outputs.hidden_states[-1][:, 0, :]
        
        # Positive representations
        pos_inputs = teacher_tokenizer(
            positives, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(target_device)
        
        pos_outputs = teacher_model(**pos_inputs, output_hidden_states=True)
        
        if hasattr(pos_outputs, 'last_hidden_state') and len(pos_outputs.last_hidden_state.shape) == 3:
            # LLM-based teacher - use mean pooling
            emb_pos = self.mean_pooling(pos_outputs.last_hidden_state, pos_inputs['attention_mask'])
            pos_hiddens = pos_outputs.hidden_states[-1]
        else:
            # BERT-based teacher - use [CLS] token
            emb_pos = pos_outputs.last_hidden_state[:, 0]
            pos_hiddens = pos_outputs.hidden_states[-1][:, 0, :]
        
        # Normalize teacher embeddings
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        # Ensure all outputs are on target device
        emb_anchor = emb_anchor.to(target_device)
        emb_pos = emb_pos.to(target_device)
        anchor_hiddens = anchor_hiddens.to(target_device)
        pos_hiddens = pos_hiddens.to(target_device)
        
        return emb_anchor, emb_pos, anchor_hiddens, pos_hiddens

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask

    def compute_mnr_loss(self, emb_anchor, emb_pos):
        """Compute Multiple Negatives Ranking Loss"""
        # Check for NaN or Inf
        if torch.isnan(emb_anchor).any() or torch.isinf(emb_anchor).any():
            print("❌ emb_anchor has NaN or Inf")
        if torch.isnan(emb_pos).any() or torch.isinf(emb_pos).any():
            print("❌ emb_pos has NaN or Inf")
        
        scores = torch.matmul(emb_anchor, emb_pos.T) * self.scale  # (B, B)
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return loss

    def compute_dual_space_kd_loss_with_cma(
        self, student_emb_anchor, student_emb_pos, student_anchor_hiddens, student_pos_hiddens,
        teacher_emb_anchor, teacher_emb_pos, teacher_anchor_hiddens, teacher_pos_hiddens,
        anchors, positives, distiller, log
    ):
        """Compute dual-space KD loss with cross-model attention for IR"""
        
        # Ensure all tensors are on the same device
        device = student_emb_anchor.device
        teacher_emb_anchor = teacher_emb_anchor.to(device)
        teacher_emb_pos = teacher_emb_pos.to(device)
        teacher_anchor_hiddens = teacher_anchor_hiddens.to(device)
        teacher_pos_hiddens = teacher_pos_hiddens.to(device)
        
        # Get batch size
        batch_size = student_emb_anchor.size(0)
        
        # Handle hidden states dimensions
        # For BERT-like models: hidden_states might be [batch, hidden_dim]
        # For LLM-like models: hidden_states might be [batch, seq_len, hidden_dim] - we need to pool
        if len(student_anchor_hiddens.shape) == 3:
            # If 3D, take mean over sequence dimension or use [CLS] token
            student_anchor_hiddens = student_anchor_hiddens.mean(dim=1)  # [batch, hidden_dim]
            student_pos_hiddens = student_pos_hiddens.mean(dim=1)
        
        if len(teacher_anchor_hiddens.shape) == 3:
            teacher_anchor_hiddens = teacher_anchor_hiddens.mean(dim=1)  # [batch, hidden_dim]
            teacher_pos_hiddens = teacher_pos_hiddens.mean(dim=1)
        
        # Get input token embeddings (first token only)
        stu_anchor_embeds, stu_pos_embeds = self.get_input_embeddings(
            distiller.student_model, anchors, positives, distiller.student_tokenizer
        )
        tea_anchor_embeds, tea_pos_embeds = self.get_input_embeddings(
            distiller.teacher_model, anchors, positives, distiller.teacher_tokenizers
        )
        
        # Ensure embedding tensors are on correct device and have correct shape
        stu_anchor_embeds = stu_anchor_embeds.to(device)  # [batch, embed_dim]
        stu_pos_embeds = stu_pos_embeds.to(device)
        tea_anchor_embeds = tea_anchor_embeds.to(device)
        tea_pos_embeds = tea_pos_embeds.to(device)
        
        # Normalize teacher representations
        norm_tea_anchor_embeds = F.normalize(tea_anchor_embeds, p=2, dim=-1)
        norm_tea_pos_embeds = F.normalize(tea_pos_embeds, p=2, dim=-1)
        norm_teacher_anchor_hiddens = F.normalize(teacher_anchor_hiddens, p=2, dim=-1)
        norm_teacher_pos_hiddens = F.normalize(teacher_pos_hiddens, p=2, dim=-1)
        
        # === Anchor Processing ===
        # CMA projections for anchors - ensure proper dimensions
        stu_q_anchor = distiller.projectors["query"](stu_anchor_embeds).float()  # [batch, proj_dim]
        tea_k_anchor = norm_tea_anchor_embeds.float()  # [batch, embed_dim]
        
        stu_v_anchor = distiller.projectors["s2t"](student_anchor_hiddens).float()  # [batch, proj_dim]
        tea_v_anchor = distiller.projectors["t2s"](norm_teacher_anchor_hiddens).float()  # [batch, proj_dim]
        
        # Compute alignment scores - batch matrix multiplication
        align_anchor = torch.bmm(
            stu_q_anchor.unsqueeze(1),  # [batch, 1, proj_dim]
            tea_k_anchor.unsqueeze(-1)  # [batch, embed_dim, 1]
        ).squeeze()  # [batch]
        
        if len(align_anchor.shape) == 0:
            align_anchor = align_anchor.unsqueeze(0)
        
        # Convert to attention weights
        align_anchor = align_anchor / (stu_q_anchor.shape[-1] ** 0.5)
        t2s_weight_anchor = torch.softmax(align_anchor.unsqueeze(-1), dim=0)  # [batch, 1]
        
        # Weighted combination for teacher-to-student
        t2s_anchor_hiddens = (t2s_weight_anchor * tea_v_anchor).to(student_anchor_hiddens.dtype)
        
        # Student-to-teacher (use normalized hidden states)
        s2t_anchor_hiddens = F.normalize(stu_v_anchor, p=2, dim=-1).to(student_anchor_hiddens.dtype)
        
        # === Positive Processing ===
        # CMA projections for positives
        stu_q_pos = distiller.projectors["query"](stu_pos_embeds).float()  # [batch, proj_dim]
        tea_k_pos = norm_tea_pos_embeds.float()  # [batch, embed_dim]
        
        stu_v_pos = distiller.projectors["s2t"](student_pos_hiddens).float()  # [batch, proj_dim]
        tea_v_pos = distiller.projectors["t2s"](norm_teacher_pos_hiddens).float()  # [batch, proj_dim]
        
        # Compute alignment scores for positives
        align_pos = torch.bmm(
            stu_q_pos.unsqueeze(1),  # [batch, 1, proj_dim]
            tea_k_pos.unsqueeze(-1)  # [batch, embed_dim, 1]
        ).squeeze()  # [batch]
        
        if len(align_pos.shape) == 0:
            align_pos = align_pos.unsqueeze(0)
        
        # Convert to attention weights
        align_pos = align_pos / (stu_q_pos.shape[-1] ** 0.5)
        t2s_weight_pos = torch.softmax(align_pos.unsqueeze(-1), dim=0)  # [batch, 1]
        
        # Weighted combination for teacher-to-student
        t2s_pos_hiddens = (t2s_weight_pos * tea_v_pos).to(student_pos_hiddens.dtype)
        
        # Student-to-teacher (use normalized hidden states)
        s2t_pos_hiddens = F.normalize(stu_v_pos, p=2, dim=-1).to(student_pos_hiddens.dtype)
        
        # === Compute Similarity Scores ===
        # Original student and teacher scores
        student_scores = torch.matmul(student_emb_anchor, student_emb_pos.T) * self.scale
        teacher_scores = torch.matmul(teacher_emb_anchor, teacher_emb_pos.T) * self.scale
        
        # t2s projected scores
        t2s_anchor_emb = F.normalize(t2s_anchor_hiddens, p=2, dim=1)
        t2s_pos_emb = F.normalize(t2s_pos_hiddens, p=2, dim=1)
        t2s_scores = torch.matmul(t2s_anchor_emb, t2s_pos_emb.T) * self.scale
        
        # s2t projected scores
        s2t_anchor_emb = F.normalize(s2t_anchor_hiddens, p=2, dim=1)
        s2t_pos_emb = F.normalize(s2t_pos_hiddens, p=2, dim=1)
        s2t_scores = torch.matmul(s2t_anchor_emb, s2t_pos_emb.T) * self.scale
        
        # === Compute Losses ===
        labels = torch.arange(student_scores.size(0), device=student_scores.device)
        
        # t2s losses
        t2s_ce_loss = F.cross_entropy(t2s_scores, labels)
        t2s_kd_loss = F.kl_div(
            F.log_softmax(student_scores, dim=-1),
            F.softmax(t2s_scores.detach(), dim=-1),
            reduction='batchmean'
        )
        
        # s2t loss
        s2t_kd_loss = F.kl_div(
            F.log_softmax(s2t_scores, dim=-1),
            F.softmax(teacher_scores, dim=-1),
            reduction='batchmean'
        )
        
        # Combine KD losses
        kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
        
        # Compute accuracies (top-1 retrieval accuracy)
        t2s_acc = (t2s_scores.argmax(-1) == labels).float().mean()
        s2t_acc = (s2t_scores.argmax(-1) == labels).float().mean()
        
        # Logging
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss
        
        return kd_loss

    def get_input_embeddings(self, model, anchors, positives, tokenizer):
        """Extract input embeddings for anchor and positive texts"""
        device = next(model.parameters()).device
        
        # Get embedding layer
        if hasattr(model, "get_input_embeddings"):
            embed_tokens = model.get_input_embeddings()
        elif hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            embed_tokens = model.bert.embeddings.word_embeddings
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embed_tokens = model.transformer.wte
        else:
            raise NotImplementedError("Unsupported model architecture for embedding extraction")
        
        # Tokenize and get embeddings
        anchor_inputs = tokenizer(
            anchors, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(device)
        
        pos_inputs = tokenizer(
            positives, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(device)
        
        # Get embeddings for first token ([CLS] or equivalent)
        # Ensure we get the correct batch dimension
        anchor_token_ids = anchor_inputs["input_ids"][:, 0]  # [batch_size]
        pos_token_ids = pos_inputs["input_ids"][:, 0]  # [batch_size]
        
        anchor_embeds = embed_tokens(anchor_token_ids).detach()  # [batch_size, embed_dim]
        pos_embeds = embed_tokens(pos_token_ids).detach()  # [batch_size, embed_dim]
        
        # Debug prints to check shapes
        print(f"Debug - anchor_embeds shape: {anchor_embeds.shape}")
        print(f"Debug - pos_embeds shape: {pos_embeds.shape}")
        
        return anchor_embeds, pos_embeds
