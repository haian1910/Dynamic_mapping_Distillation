import torch
import torch.nn as nn
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scale = 1.0
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()

    # def forward(self, emb_anchor, emb_pos):
    #     # emb_anchor: (B, D), emb_pos: (B, D)
    #     scores = torch.matmul(emb_anchor, emb_pos.T) * self.scale  # (B, B)
    #     labels = torch.arange(scores.size(0), device=scores.device)
    #     return self.cross_entropy(scores, labels)

    def forward(self, distiller, anchors, positives, logging_output, batch_denom):
        """
        Compute MultipleNegativesRankingLoss tasks.
        - anchors list of queries
        - positives is list of positive documents.
        """
        self.distiller = distiller
        student_model = distiller.student_model
        student_tokenizer = distiller.student_tokenizer
        
        if self.args.peft: #student is llm2vec. i.e sft teacher
            #anchors
            inputs = student_tokenizer(anchors, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
            outputs = student_model(**inputs)

            token_embeddings = outputs.last_hidden_state           # [B, T, D]
            attention_mask = inputs['attention_mask']              # [B, T]

            # Expand mask: [B, T] → [B, T, 1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

            # Masked sum then divide by actual token count
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)        # [B, D]
            sum_mask = input_mask_expanded.sum(dim=1)                                         # [B, 1]
            emb_anchor = sum_embeddings / sum_mask                                           # [B, D]

            # import pdb; pdb.set_trace()

            ## postives
            inputs = student_tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
            outputs = student_model(**inputs)

            token_embeddings = outputs.last_hidden_state           # [B, T, D]
            attention_mask = inputs['attention_mask']              # [B, T]

            # Expand mask: [B, T] → [B, T, 1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

            # Masked sum then divide by actual token count
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)        # [B, D]
            sum_mask = input_mask_expanded.sum(dim=1)                                         # [B, 1]
            emb_pos = sum_embeddings / sum_mask                                           # [B, D]

        else: #student is bert
            anchor_inputs = student_tokenizer(anchors, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
            outputs = student_model(**anchor_inputs)  # BERTModel outputs
            emb_anchor = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

            positives_inputs = student_tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(student_model.device)
            outputs = student_model(**positives_inputs)  # BERTModel outputs
            emb_pos = outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

            emb_anchor = torch.nn.functional.normalize(emb_anchor, p=2, dim=1)
            emb_pos    = torch.nn.functional.normalize(emb_pos, p=2, dim=1)
            # import pdb; pdb.set_trace()


        if torch.isnan(emb_anchor).any() or torch.isinf(emb_anchor).any():
            print("❌ emb_anchor has NaN or Inf")
        if torch.isnan(emb_pos).any() or torch.isinf(emb_pos).any():
            print("❌ emb_pos has NaN or Inf")


        scores = torch.matmul(emb_anchor, emb_pos.T) * self.scale  # (B, B)
        labels = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, labels)

        # loss = self.cross_entropy(scores, labels)
        # print(f'==REMOVEME loss labels {labels}, loss {loss}')


        # Update logging output, return to main distillation
        # logging_output = self.record_logging_output(
        #     logging_output, 
        #     batch_denom, {"loss": loss}
        # )
        
        return loss, {}
    
    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and correlations for logging, handling distributed training.
        """

        for k, v in content.items():            
            record_v = v / batch_denom
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output


class MultipleNegativeRankingLoss(nn.Module):
    def __init__(self, args) -> None:
        super(STSLoss, self).__init__()
        self.loss_type = getattr(args, "sts_loss_type", "mse")  # Default to MSE if not specified
        # Check if BF16 is being used
        self.use_bf16 = getattr(args, "bf16", False)
        if not hasattr(args, "bf16"):
            # Try to detect from DeepSpeed config
            if hasattr(args, "deepspeed_config"):
                try:
                    import json
                    with open(args.deepspeed_config, 'r') as f:
                        ds_config = json.load(f)
                        self.use_bf16 = ds_config.get("bf16", {}).get("enabled", False)
                except:
                    pass
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute loss for STS (Semantic Textual Similarity) tasks.
        - Expects model output (batch_size, 1), target (batch_size, 1).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]

        # Convert target to BF16 if needed
        if self.use_bf16 and target.dtype == torch.float32:
            target = target.to(torch.bfloat16)

        # Forward pass through the model
        model_output = model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
            token_type_ids=input_data['token_type_ids'] if 'token_type_ids' in input_data else None
        )
        
        predictions = model_output.scores
            
        # Ensure predictions are the right shape
        if predictions.shape[-1] != 1:
            # If the model outputs multiple values, use mean pooling or a linear layer
            if hasattr(self.distiller, "regression_head"):
                predictions = self.distiller.regression_head(predictions)
            else:
                # Create a simple regression head on first use
                self.distiller.regression_head = nn.Linear(predictions.size(-1), 1).to(predictions.device)
                if self.use_bf16:
                    self.distiller.regression_head = self.distiller.regression_head.to(torch.bfloat16)
                self.distiller.regression_head.weight.data.normal_(mean=0.0, std=0.02)
                self.distiller.regression_head.bias.data.zero_()
                predictions = self.distiller.regression_head(predictions)

        # Ensure consistent dtype
        if self.use_bf16:
            if predictions.dtype != torch.bfloat16:
                predictions = predictions.to(torch.bfloat16)
            if target.dtype != torch.bfloat16:
                target = target.to(torch.bfloat16)

        # Compute loss based on specified type
        loss = self.compute_sts_loss(predictions, target)
        
        # Compute correlation for evaluation
        pearson, spearman = self.compute_correlations(predictions, target)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "pearson": pearson,
                "spearman": spearman,
                "predictions": predictions.detach().mean(),
                "target": target.detach().mean()
            }
        )
        return loss, logging_output

    def compute_sts_loss(self, predictions, target):
        if self.loss_type == "mse":
            return nn.MSELoss()(predictions, target)
        elif self.loss_type == "mae":
            return nn.L1Loss()(predictions, target)
        elif self.loss_type == "huber":
            return nn.SmoothL1Loss()(predictions, target)
        else:
            # Default to MSE
            return nn.MSELoss()(predictions, target)

    def compute_correlations(self, predictions, target):
        """
        Compute Pearson and Spearman correlations between predictions and targets.
        Returns naive local estimates (proper correlation will be computed at evaluation).
        """
        # Detach and convert to CPU and float32 for correlation calculation
        pred_flat = predictions.detach().to(torch.float32).view(-1).cpu()
        target_flat = target.detach().to(torch.float32).view(-1).cpu()
        
        # Calculate mean and standard deviation for Pearson correlation
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        
        # Simple Pearson correlation estimate (for logging only)
        try:
            # Calculate Pearson correlation coefficient
            numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
            denominator = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() * ((target_flat - target_mean) ** 2).sum())
            pearson = numerator / denominator if denominator != 0 else torch.tensor(0.0)
            
            # Simple placeholder for Spearman (proper calculation requires rank conversion)
            # In practice, this would be calculated during evaluation, not training
            spearman = pearson  # Placeholder
        except:
            # Handle potential numerical issues
            pearson = torch.tensor(0.0)
            spearman = torch.tensor(0.0)
            
        return pearson, spearman

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and correlations for logging, handling distributed training.
        """
        # Get the device of the current process
        if torch.distributed.is_initialized():
            # Get the device used in the current process
            device_params = list(self.parameters())
            if device_params:
                device = device_params[0].device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        for k, v in content.items():
            if k in ["pearson", "spearman"]:
                # For correlations, we don't need to divide by batch_denom
                # but we still average across processes
                if isinstance(v, torch.Tensor):
                    # Move tensor to correct device before all_reduce
                    # Always use float32 for these statistics
                    record_v = v.clone().to(device).to(torch.float32)
                    if torch.distributed.is_initialized():
                        try:
                            dist.all_reduce(record_v, dist.ReduceOp.SUM)
                            record_v = record_v.item() / dist.get_world_size()
                        except RuntimeError:
                            # Fallback if all_reduce fails
                            record_v = v.item()
                    else:
                        record_v = v.item()
                else:
                    record_v = v
                    if torch.distributed.is_initialized():
                        record_v = record_v / dist.get_world_size()
            elif k in ["predictions", "target"]:
                # Just record mean values for monitoring
                if isinstance(v, torch.Tensor):
                    record_v = v.item()
                else:
                    record_v = v
            else:
                # Normalize loss by batch_denom and average across processes
                if isinstance(v, torch.Tensor):
                    # Move tensor to correct device before all_reduce
                    # Always use float32 for these statistics
                    record_v = (v / batch_denom).to(torch.float32)
                    if torch.distributed.is_initialized():
                        record_v = record_v.to(device)
                        try:
                            dist.all_reduce(record_v, dist.ReduceOp.SUM)
                            record_v = record_v.item() / dist.get_world_size()
                        except RuntimeError:
                            # Fallback if all_reduce fails
                            record_v = (v / batch_denom).item()
                    else:
                        record_v = record_v.item()
                else:
                    record_v = v / batch_denom
                    if torch.distributed.is_initialized():
                        record_v = record_v / dist.get_world_size()
                    
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
                
        return logging_output
