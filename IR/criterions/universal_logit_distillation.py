import torch
import torch.nn.functional as F
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss

class UniversalLogitDistillation(MultipleNegativesRankingLoss):
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
        Compute Universal Logit Distillation for Information Retrieval tasks.
        - anchors: list of queries
        - positives: list of positive documents
        """
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        student_tokenizer = distiller.student_tokenizer
        teacher_tokenizer = distiller.teacher_tokenizers
        
        log = {}
        
        # Get student embeddings
        student_emb_anchor, student_emb_pos = self.get_student_embeddings(
            anchors, positives, student_model, student_tokenizer
        )
        
        # Compute base Multiple Negatives Ranking Loss
        base_loss = self.compute_mnr_loss(student_emb_anchor, student_emb_pos)
        
        # Get teacher embeddings (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_emb_anchor, teacher_emb_pos = self.get_teacher_embeddings(
                anchors, positives, teacher_model, teacher_tokenizer
            )
        
        # Compute distillation loss
        kd_loss = self.compute_universal_logit_distillation_loss(
            student_emb_anchor, student_emb_pos, 
            teacher_emb_anchor, teacher_emb_pos, 
            log
        )
        
        print("uld_loss:", kd_loss)
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * base_loss + self.kd_rate * kd_loss
        log["loss"] = loss
        log["base_loss"] = base_loss
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        
        return loss, logging_output

    def get_student_embeddings(self, anchors, positives, student_model, student_tokenizer):
        """Get embeddings from student model"""
        if self.args.peft:  # student is llm2vec (SFT teacher)
            # Anchor embeddings
            inputs = student_tokenizer(
                anchors, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(student_model.device)
            
            outputs = student_model(**inputs)
            emb_anchor = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Positive embeddings
            inputs = student_tokenizer(
                positives, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(student_model.device)
            
            outputs = student_model(**inputs)
            emb_pos = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
        else:  # student is BERT
            # Anchor embeddings
            anchor_inputs = student_tokenizer(
                anchors, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(student_model.device)
            
            outputs = student_model(**anchor_inputs)
            emb_anchor = outputs.last_hidden_state[:, 0]  # [CLS] token
            
            # Positive embeddings
            positives_inputs = student_tokenizer(
                positives, padding=True, truncation=True, 
                return_tensors="pt", max_length=self.args.max_length
            ).to(student_model.device)
            
            outputs = student_model(**positives_inputs)
            emb_pos = outputs.last_hidden_state[:, 0]  # [CLS] token
            
            # Normalize embeddings
            emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
            emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        return emb_anchor, emb_pos

    def get_teacher_embeddings(self, anchors, positives, teacher_model, teacher_tokenizer):
        """Get embeddings from teacher model"""
        # Ensure teacher model is on the correct device
        device = next(teacher_model.parameters()).device
        
        # If teacher model is on CPU, move it to GPU
        if device.type == 'cpu' and torch.cuda.is_available():
            teacher_model = teacher_model.cuda()
            device = next(teacher_model.parameters()).device
        
        # Anchor embeddings
        anchor_inputs = teacher_tokenizer(
            anchors, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(device)
        
        outputs = teacher_model(**anchor_inputs)
        
        # Check if teacher is LLM-based or BERT-based
        if hasattr(outputs, 'last_hidden_state') and len(outputs.last_hidden_state.shape) == 3:
            # LLM-based teacher - use mean pooling
            emb_anchor = self.mean_pooling(outputs.last_hidden_state, anchor_inputs['attention_mask'])
        else:
            # BERT-based teacher - use [CLS] token
            emb_anchor = outputs.last_hidden_state[:, 0]
        
        # Positive embeddings
        positives_inputs = teacher_tokenizer(
            positives, padding=True, truncation=True, 
            return_tensors="pt", max_length=self.args.max_length
        ).to(device)
        
        outputs = teacher_model(**positives_inputs)
        
        if hasattr(outputs, 'last_hidden_state') and len(outputs.last_hidden_state.shape) == 3:
            # LLM-based teacher - use mean pooling
            emb_pos = self.mean_pooling(outputs.last_hidden_state, positives_inputs['attention_mask'])
        else:
            # BERT-based teacher - use [CLS] token
            emb_pos = outputs.last_hidden_state[:, 0]
        
        # Normalize teacher embeddings
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        return emb_anchor, emb_pos

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

    def compute_universal_logit_distillation_loss(
        self, student_emb_anchor, student_emb_pos, 
        teacher_emb_anchor, teacher_emb_pos, log
    ):
        """
        Compute Universal Logit Distillation loss for IR task.
        Instead of using logits, we use similarity scores between embeddings.
        """
        # Ensure all embeddings are on the same device
        device = student_emb_anchor.device
        teacher_emb_anchor = teacher_emb_anchor.to(device)
        teacher_emb_pos = teacher_emb_pos.to(device)
        
        # Compute similarity matrices (these act as our "logits" for IR)
        student_scores = torch.matmul(student_emb_anchor, student_emb_pos.T) * self.scale  # [B, B]
        teacher_scores = torch.matmul(teacher_emb_anchor, teacher_emb_pos.T) * self.scale  # [B, B]
        
        # Convert scores to probabilities
        student_probs = torch.softmax(student_scores, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_scores, dim=-1, dtype=torch.float32)
        
        # Universal Logit Distillation: absolute difference between sorted probabilities
        sorted_student_probs = student_probs.sort(dim=-1, descending=True).values
        sorted_teacher_probs = teacher_probs.sort(dim=-1, descending=True).values
        
        # Compute loss as mean absolute difference across the batch
        uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().mean()
        log["uld_loss"] = uld_loss
        
        return uld_loss
