import torch
from .multiple_negatives_ranking_loss import MultipleNegativesRankingLoss
import torch.nn as nn
import math
import editdistance
from transformers import AutoTokenizer, AutoConfig, AutoModel
import re

class OT_PRO_RMSE_CKA(MultipleNegativesRankingLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.sinkhorn_alpha = 0.1
        self.stopThr = 1e-9
        self.OT_max_iter = 100
        self.epsilon = 1e-9
        self.ot_dist_type = 'attention'
        self.importance_scaling = 0.5
    
    def forward(
        self, 
        distiller, 
        anchors,
        positives, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizers

        log = {}

        # Compute base Multiple Negatives Ranking Loss
        base_loss, base_log = super().forward(distiller, anchors, positives, logging_output, batch_denom)
        
        # Process anchors and positives for knowledge distillation
        anchor_cka_loss = self.compute_cka_loss_for_texts(anchors, student_model, teacher_model, tokenizer_student, tokenizer_teacher)
        positive_cka_loss = self.compute_cka_loss_for_texts(positives, student_model, teacher_model, tokenizer_student, tokenizer_teacher)
        
        anchor_ot_loss = self.compute_ot_loss_for_texts(anchors, student_model, teacher_model, tokenizer_student, tokenizer_teacher, distiller)
        positive_ot_loss = self.compute_ot_loss_for_texts(positives, student_model, teacher_model, tokenizer_student, tokenizer_teacher, distiller)
        
        # Average the losses
        avg_cka_loss = (anchor_cka_loss + positive_cka_loss) / 2.0
        avg_ot_loss = (anchor_ot_loss + positive_ot_loss) / 2.0
        
        print("cka_loss:", avg_cka_loss)
        print("ot_loss:", avg_ot_loss)
        print("base_loss:", base_loss)
        
        # Combine losses
        total_loss = (1.0 - self.kd_rate) * base_loss + self.kd_rate * (0.1 * avg_cka_loss + avg_ot_loss)
        
        log["loss"] = total_loss
        log["base_loss"] = base_loss
        log["cka_loss"] = avg_cka_loss
        log["ot_loss"] = avg_ot_loss

        return total_loss, log

    def compute_cka_loss_for_texts(self, texts, student_model, teacher_model, tokenizer_student, tokenizer_teacher):
        """
        Compute CKA loss for a list of texts
        """
        total_cka_loss = 0.0
        device = student_model.device
        
        # Tokenize all texts
        student_inputs = tokenizer_student(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(device)
        teacher_inputs = tokenizer_teacher(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(device)
        
        # Get model outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs, output_hidden_states=True, output_attentions=True)
        
        student_outputs = student_model(**student_inputs, output_hidden_states=True, output_attentions=True)
        
        # Process each text in the batch
        for i in range(len(texts)):
            text = texts[i].lower()
            text = re.sub(r'[^\w\s]', '', text)
            
            cka_loss = self.compute_cka_for_single_text(
                text, student_outputs, teacher_outputs, i, 
                tokenizer_student, tokenizer_teacher, device,
                student_inputs, teacher_inputs  # Pass the inputs for bounds checking
            )
            total_cka_loss += cka_loss
            
        return total_cka_loss / len(texts) if len(texts) > 0 else torch.tensor(0.0, device=device)

    def compute_cka_for_single_text(self, text, student_outputs, teacher_outputs, batch_idx, 
                                  tokenizer_student, tokenizer_teacher, device, 
                                  student_inputs, teacher_inputs):
        """
        Compute CKA loss for a single text with bounds checking
        """
        # Token mapping and alignment logic
        reciprocal_mapping = self.get_top_k_reciprocal_mapping(text, tokenizer_student, tokenizer_teacher, teacher_outputs, batch_idx)
        teacher_indices, student_indices = self.get_indices_from_mapping(text, reciprocal_mapping, tokenizer_student, tokenizer_teacher)
        
        if len(teacher_indices) == 0 or len(student_indices) == 0:
            return torch.tensor(0.0, device=device)
        
        # Get attention weights
        teacher_atts = teacher_outputs.attentions
        student_atts = student_outputs.attentions
        
        # Layer mapping
        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
        layers_per_block = teacher_layer_num // student_layer_num
        
        new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]
        
        # Use last 2 layers
        teacher_last_k_layers = new_teacher_atts[-2:]
        student_last_k_layers = student_atts[-2:]
        
        cka_loss_fn = CKALoss(eps=1e-8).to(device)
        total_cka_loss = 0.0
        
        # Get sequence lengths for bounds checking
        teacher_seq_len = teacher_inputs['attention_mask'][batch_idx].sum().item()
        student_seq_len = student_inputs['attention_mask'][batch_idx].sum().item()
        
        for teacher_att, student_att in zip(teacher_last_k_layers, student_last_k_layers):
            # Get attention tensor dimensions
            teacher_att_shape = teacher_att.shape  # [batch, heads, seq_len, seq_len]
            student_att_shape = student_att.shape
            
            # Filter indices to be within bounds
            valid_teacher_indices = [idx for idx in teacher_indices if idx < teacher_att_shape[2] and idx < teacher_seq_len]
            valid_student_indices = [idx for idx in student_indices if idx < student_att_shape[2] and idx < student_seq_len]
            
            if len(valid_teacher_indices) == 0 or len(valid_student_indices) == 0:
                continue
            
            # Extract attention for selected tokens with bounds checking
            try:
                teacher_att_for_k_token = teacher_att[batch_idx, :, valid_teacher_indices, :].mean(dim=0)
                student_att_for_k_token = student_att[batch_idx, :, valid_student_indices, :].mean(dim=0)
                
                # Ensure we don't exceed sequence length in the last dimension
                max_teacher_len = min(teacher_att_for_k_token.shape[1], teacher_seq_len)
                max_student_len = min(student_att_for_k_token.shape[1], student_seq_len)
                
                teacher_att_for_k_token = teacher_att_for_k_token[:, :max_teacher_len]
                student_att_for_k_token = student_att_for_k_token[:, :max_student_len]
                
                # Handle small values
                teacher_att_for_k_token = torch.where(
                    teacher_att_for_k_token <= -1e2,
                    torch.zeros_like(teacher_att_for_k_token).to(device),
                    teacher_att_for_k_token
                )
                student_att_for_k_token = torch.where(
                    student_att_for_k_token <= -1e2,
                    torch.zeros_like(student_att_for_k_token).to(device),
                    student_att_for_k_token
                )
                
                # Only compute CKA if both tensors have valid dimensions
                if teacher_att_for_k_token.numel() > 0 and student_att_for_k_token.numel() > 0:
                    cka_loss = cka_loss_fn(student_att_for_k_token, teacher_att_for_k_token)
                    total_cka_loss += cka_loss

            except RuntimeError as e:
                print(f"Error in attention extraction: {e}")
                print(f"Teacher indices: {valid_teacher_indices}, Student indices: {valid_student_indices}")
                print(f"Teacher att shape: {teacher_att_shape}, Student att shape: {student_att_shape}")
                continue
            
        return total_cka_loss

    def compute_ot_loss_for_texts(self, texts, student_model, teacher_model, tokenizer_student, tokenizer_teacher, distiller):
        """
        Compute OT loss for a list of texts
        """
        total_ot_loss = 0.0
        device = student_model.device
        
        # Tokenize all texts
        student_inputs = tokenizer_student(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(device)
        teacher_inputs = tokenizer_teacher(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_length).to(device)
        
        # Get model outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs, output_hidden_states=True, output_attentions=True)
        
        student_outputs = student_model(**student_inputs, output_hidden_states=True, output_attentions=True)
        
        # Process each text in the batch
        for i in range(len(texts)):
            ot_loss = self.compute_ot_for_single_text(
                texts[i], student_outputs, teacher_outputs, i,
                student_inputs, teacher_inputs, distiller
            )
            total_ot_loss += ot_loss
            
        return total_ot_loss / len(texts) if len(texts) > 0 else torch.tensor(0.0, device=device)

    def compute_ot_for_single_text(self, text, student_outputs, teacher_outputs, batch_idx, student_inputs, teacher_inputs, distiller):
        """
        Compute OT loss for a single text
        """
        device = student_outputs.hidden_states[-1].device
        
        # Get hidden states
        student_features = student_outputs.hidden_states[-1][batch_idx]  # (seq_len, hidden_dim)
        teacher_features = teacher_outputs.hidden_states[-1][batch_idx]  # (seq_len, hidden_dim)
        
        # Get attention masks
        student_mask = student_inputs['attention_mask'][batch_idx]
        teacher_mask = teacher_inputs['attention_mask'][batch_idx]
        
        # Get valid sequences
        valid_student_len = int(student_mask.sum().item())
        valid_teacher_len = int(teacher_mask.sum().item())
        
        valid_student_seq = student_features[:valid_student_len]
        valid_teacher_seq = teacher_features[:valid_teacher_len]
        
        # Project teacher features to student space
        if hasattr(distiller, 'projectors') and 't2s' in distiller.projectors:
            projector = distiller.projectors["t2s"]
            projected_teacher_seq = projector(valid_teacher_seq)
        else:
            # If no projector, assume same dimensionality
            projected_teacher_seq = valid_teacher_seq
        
        # Ensure same dtype
        projected_teacher_seq = projected_teacher_seq.to(valid_student_seq.dtype)
        
        # Create uniform mass distributions
        tea_mass = torch.ones(valid_teacher_seq.size(0), 1, device=device, dtype=torch.float32) / valid_teacher_seq.size(0)
        stu_mass = torch.ones(valid_student_seq.size(0), 1, device=device, dtype=torch.float32) / valid_student_seq.size(0)
        
        # Convert sequences to float32 for computation
        valid_student_seq = valid_student_seq.to(torch.float32)
        projected_teacher_seq = projected_teacher_seq.to(torch.float32)
        
        # Compute cost matrix
        if self.ot_dist_type == 'euclidean':
            cost_matrix = self.pairwise_euclidean_distance(valid_student_seq, projected_teacher_seq)
        elif self.ot_dist_type == 'cosine':
            cost_matrix = self.pairwise_cosine_distance(valid_student_seq, projected_teacher_seq)
        elif self.ot_dist_type == 'attention':
            cost_matrix = self.pairwise_attention_distance(valid_student_seq, projected_teacher_seq)
        else:
            raise ValueError(f"Unknown distance type: {self.ot_dist_type}")
        
        cost_matrix = cost_matrix.to(torch.float32)
        
        # Compute OT loss
        ot_loss, _ = self.sinkhorn(cost_matrix, stu_mass, tea_mass)
        
        return ot_loss

    # Helper methods (keeping the original implementations with some fixes)
    def preprocess_text(self, text):
        # Remove numbers if specified
        text = re.sub(r'\d+', '', text)

        # Custom list of English stopwords (a common subset)
        stop_words = [
            'a', 'an', 'the', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through',
            'during', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'other', 'such',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'would', 'could', 'should', 'ought', 'i\'m', 'you\'re', 'he\'s',
            'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve',
            'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll',
            'she\'ll', 'we\'ll', 'they\'ll', 'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s', 'when\'s', 'where\'s',
            'why\'s', 'how\'s', '.'
        ]

        words = [word for word in text.split() if word not in stop_words]
        text = ' '.join(words)
        return text

    def find_best_mapping(self, x, base_tokens, blending_special, base_special, best_one=True):
        tmp_x = x.replace(blending_special, base_special)
        if tmp_x in base_tokens:
            return tmp_x, tmp_x
        else:
            if best_one:
                best = None
                best_dist = None
                for y in base_tokens:
                    d = editdistance.eval(tmp_x, y)
                    if best is None or d < best_dist:
                        best = y
                        best_dist = d
                return tmp_x, best
            else:
                token_and_distance = [(y, editdistance.eval(tmp_x, y)) for y in base_tokens]
                min_distance = min(d for _, d in token_and_distance)
                shortest_distance_tokens = [y for y, d in token_and_distance if d == min_distance]
                return tmp_x, shortest_distance_tokens

    def align_text_tokens(self, text, tokenizer_student, tokenizer_teacher):
        teacher_special = "<s>"
        student_special = "[CLS]"
        
        teacher_tokens = set(tokenizer_teacher.tokenize(text))
        student_tokens = set(tokenizer_student.tokenize(text))

        teacher_to_student = {}
        for t in teacher_tokens:
            _, s = self.find_best_mapping(t, student_tokens, teacher_special, student_special, best_one=True)
            teacher_to_student[t] = s

        student_to_teacher = {}
        for s in student_tokens:
            _, t = self.find_best_mapping(s, teacher_tokens, student_special, teacher_special, best_one=True)
            student_to_teacher[s] = t

        reciprocal_mapping = {}
        for t, s in teacher_to_student.items():
            if s in student_to_teacher and student_to_teacher[s] == t:
                reciprocal_mapping[t] = s

        return reciprocal_mapping

    def get_indices_from_mapping(self, text, reciprocal_mapping, tokenizer_student, tokenizer_teacher):
        input_ids_teacher = tokenizer_teacher.encode(text, return_tensors='pt')[0]
        input_ids_student = tokenizer_student.encode(text, return_tensors='pt')[0]
        
        teacher_token_ids = {tokenizer_teacher.convert_tokens_to_ids(t) for t in reciprocal_mapping.keys()}
        student_token_ids = {tokenizer_student.convert_tokens_to_ids(s) for s in reciprocal_mapping.values()}
        
        teacher_indices = []
        seen_teacher = set()
        for idx, token_id in enumerate(input_ids_teacher):
            tid = token_id.item()
            if tid in teacher_token_ids and tid not in seen_teacher:
                teacher_indices.append(idx)
                seen_teacher.add(tid)
        
        student_indices = []
        seen_student = set()
        for idx, token_id in enumerate(input_ids_student):
            tid = token_id.item()
            if tid in student_token_ids and tid not in seen_student:
                student_indices.append(idx)
                seen_student.add(tid)
        
        return teacher_indices, student_indices

    def extract_top_k_tokens(self, text, k, teacher_model, tokenizer_teacher):
        text = self.preprocess_text(text)
        tokenizer = tokenizer_teacher

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(teacher_model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = teacher_model(**inputs, output_hidden_states=True, output_attentions=True)

        last_layer_attention = outputs.attentions[-1].squeeze(0)
        avg_attention = last_layer_attention.mean(dim=0)
        token_importance = avg_attention.sum(dim=0).to(torch.float32).cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_importance_pairs = list(zip(tokens, token_importance))
        top_k_tokens = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)[:k]

        return top_k_tokens

    def get_top_k_reciprocal_mapping(self, text, tokenizer_student, tokenizer_teacher, teacher_outputs, batch_idx):
        reciprocal_mapping = self.align_text_tokens(text, tokenizer_student, tokenizer_teacher)
        n = len(reciprocal_mapping)
        
        if n == 0:
            return {}
        
        # Use teacher_outputs directly instead of calling the model again
        last_layer_attention = teacher_outputs.attentions[-1][batch_idx]
        avg_attention = last_layer_attention.mean(dim=0)
        token_importance = avg_attention.sum(dim=0).to(torch.float32).cpu().numpy()

        tokens = tokenizer_teacher.convert_ids_to_tokens(tokenizer_teacher.encode(text, return_tensors='pt')[0])
        
        # Ensure token_importance and tokens have the same length
        min_len = min(len(tokens), len(token_importance))
        tokens = tokens[:min_len]
        token_importance = token_importance[:min_len]
        
        token_importance_pairs = list(zip(tokens, token_importance))
        top_k_tokens = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)[:max(1, n//3)]
        top_k_tokens_set = {token for token, _ in top_k_tokens}
        
        reciprocal_mapping_top_k = {t: s for t, s in reciprocal_mapping.items() if t in top_k_tokens_set}
        return reciprocal_mapping_top_k

    # Keep the original distance computation methods
    def pairwise_euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def pairwise_cosine_distance(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=a.dtype))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=b.dtype))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        sim_mt = 1 - sim_mt
        return sim_mt

    def pairwise_attention_distance(self, x, y, eps=1e-8):
        d = x.shape[1]
        sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
        attention_weights = torch.softmax(sim_mt, dim=1)
        dist_mt = 1.0 - attention_weights
        return dist_mt

    def sinkhorn(self, cost_matrix, a, b, num_iters=None):
        if num_iters is None:
            num_iters = self.OT_max_iter
        
        m, n = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        
        if m == 0 or n == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), torch.zeros((m, n), device=device, dtype=dtype)
        
        if a.dim() == 1:
            a = a.view(-1, 1)
        if b.dim() == 1:
            b = b.view(-1, 1)
            
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        
        if a.shape[0] != m:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
        if b.shape[0] != n:
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        
        if torch.sum(a) < self.epsilon or torch.sum(b) < self.epsilon:
            a = torch.ones(m, 1, device=device, dtype=dtype) / m
            b = torch.ones(n, 1, device=device, dtype=dtype) / n
        else:
            a = a / torch.sum(a)
            b = b / torch.sum(b)
        
        K = torch.exp(-cost_matrix / self.sinkhorn_alpha)
        u = torch.ones(m, 1, device=device, dtype=dtype)
        v = torch.ones(n, 1, device=device, dtype=dtype)
        
        for _ in range(num_iters):
            u_prev = u.clone()
            
            KTu = torch.matmul(K.t(), u)
            v = b / (KTu + self.epsilon)
            
            Kv = torch.matmul(K, v)
            u = a / (Kv + self.epsilon)
            
            err = torch.norm(u - u_prev, p=float('inf'))
            if err < self.stopThr:
                break
        
        P = torch.diag(u.squeeze()) @ K @ torch.diag(v.squeeze())
        ot_loss = torch.sum(P * cost_matrix)
        
        return ot_loss, P


class CKALoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num / torch.sqrt(den1 * den2)
