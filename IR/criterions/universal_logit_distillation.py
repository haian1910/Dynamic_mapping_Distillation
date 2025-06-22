import torch
import torch.nn.functional as F

class UniversalLogitDistillation:
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
    

    def forward(self, distiller, anchors, positives, logging_output, batch_denom):
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        
        #student forward pass to compute the loss
        anchor_inputs = student_tokenizer(anchors, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
        outputs = student_model(**anchor_inputs)  # BERTModel outputs
        emb_anchor = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

        positives_inputs = student_tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
        outputs = student_model(**positives_inputs)  # BERTModel outputs
        emb_pos = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

        emb_anchor = torch.nn.functional.normalize(emb_anchor, p=2, dim=1)
        emb_pos    = torch.nn.functional.normalize(emb_pos, p=2, dim=1)


        scores = torch.matmul(emb_anchor, emb_pos.T) * self.scale  # (B, B)
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)


        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute distillation loss
        kd_loss, log = self.compute_universal_logit_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        print("uld_loss:", kd_loss)
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy
        # accuracy = self.compute_accuracy(
        #     logits, output_data["labels"]
        # )
        # log["accuracy"] = accuracy

        # # Update logging output
        # logging_output = self.record_logging_output(
        #     logging_output, batch_denom, log
        # )
        return loss, {}

    def compute_universal_logit_distillation_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_logits = outputs.logits  # [batch_size, num_classes]
        teacher_logits = teacher_outputs.logits  # [batch_size, num_classes]

        # Handle potential mismatch in number of classes (vocab size)
        vocab_size_gap = student_logits.shape[-1] - teacher_logits.shape[-1]
        if vocab_size_gap > 0:
            # Pad teacher logits with zeros if student has more classes
            teacher_logits = torch.cat(
                [teacher_logits, torch.zeros_like(student_logits[:, :vocab_size_gap])], 
                dim=-1
            )
        elif vocab_size_gap < 0:
            # Pad student logits with zeros if teacher has more classes
            student_logits = torch.cat(
                [student_logits, torch.zeros_like(teacher_logits[:, :(-vocab_size_gap)])], 
                dim=-1
            )

        # Compute softened probabilities
        student_probs = torch.softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, dim=-1, dtype=torch.float32)

        # Universal Logit Distillation loss: absolute difference between sorted probabilities
        sorted_student_probs = student_probs.sort(dim=-1, descending=True).values
        sorted_teacher_probs = teacher_probs.sort(dim=-1, descending=True).values
        
        # Compute loss as mean absolute difference across the batch
        uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().mean()
        log["uld_loss"] = uld_loss
        
        return uld_loss, log
