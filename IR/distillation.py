import time
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers.util import cos_sim #da co normalize roi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn.functional import normalize
import deepspeed
import shutil
import json
from tqdm import tqdm
from tqdm import trange
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModel,
)
from transformers.integrations import HfDeepSpeedConfig
from IR.arguments import get_args
from IR.distiller import Distiller
from IR.data_utils.data_set import IRDataset
from IR.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from IR.criterions import build_criterion

torch.set_num_threads(4) # limit the number of threads used by torch for cpu




def prepare_dataset(args, distiller):
    data = {}
    data_path = args.data_dir
    
    if args.do_train:
        data['train'] = IRDataset(args, data_path)
        data['train'].load(split="train")
        
        data['dev'] = IRDataset(args, data_path)
        data['dev'].load(split="dev")

        data['test'] = IRDataset(args, data_path)
        data['test'].load(split="test")

    elif args.do_eval:
        data['test'] = IRDataset(args, data_path)
        data['test'].load(split="test")

    else:
        raise ValueError("Do train and do eval must set one")
        
    return data

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    log_rank(f"dp_world_size {dp_world_size}, dp_rank {dp_rank}")

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    model_list = []
    logging_output = {
        "epoch": 0,
        "global_step": 1,
        "loss": [], 
        "pearson": [],
        "spearman": [],
        "kd_loss": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        print("Training mode?", model.student_model.training)  # True

        epoch_start_time = time.time()
        step = 0
        total_samples = 0
        total_time = 0.0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        
        

        for batch in data_iter:
            st_time = time.time()
            anchor, positive = batch
            # dataset["train"].move_to_device([input_batch, output_batch], device)
            
            model.zero_grad()
            loss, _ = model(
                criterion,
                anchor,
                positive,
                logging_output,
                loss_denom=1, #deepspeed support sync gradient, no need to calculate loss_denom
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            model.backward(loss)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if grad_norm == 0 or torch.isnan(grad_norm):
                # print(f"⚠️ Grad norm = {grad_norm}, skipping step.")
                model.zero_grad()
                continue

            model.step()
            # torch.cuda.synchronize()  # correctly compute time

            elapsed_time = time.time() - st_time
            num_samples = len(anchor)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1

            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)

            if logging_output["global_step"]%10 == 0:
                print("Allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
                print("Reserved :", torch.cuda.memory_reserved() / 1e6, "MB")


            if dist.get_rank() == 0:
                data_iter.set_postfix(loss=loss.item())


        #     if args.save_dir and logging_output["global_step"]%1000 == 0 and dist.get_rank() == 0: #save_interval = 1 then save each epoch
        # # if args.save_dir and (epoch+1)%args.save_interval == 0 and dist.get_rank() == 0: #save_interval = 1 then save each epoch
        #     #eval_interval = 1 then evaluate each epoch
        #         log_rank(f"Evaluating before saving model... epoch {epoch} step {logging_output['global_step']}")
        #         eval_mrr, eval_ndcg, eval_map = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
        #         # if "test" in dataset: #evaluate for test, no affect
        #         #     _, _, _ = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
        #         ckpt_name = f"epoch{epoch + 1}_step{logging_output['global_step']}_mrr_{eval_mrr:.4f}_map{eval_map:.4f}_ndcg{eval_ndcg:.4f}"
        #         save_dir_path = os.path.join(args.save_dir, ckpt_name)
                
        #         os.makedirs(save_dir_path, exist_ok=True)
        #         if not args.only_save_projector:
        #             log_rank("Saving tokenizer...")
        #             tokenizer.save_pretrained(save_dir_path)
        #             log_rank("Saving model...")
        #             model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
        #             log_rank("Saving config")
        #             model.module.student_model.config.save_pretrained(save_dir_path)
                
        #         if hasattr(model.module, "projectors"):
        #             log_rank("Saving projector...")
        #             torch.save(
        #                 model.module.projectors.state_dict(), 
        #                 os.path.join(save_dir_path, "projector.pt")
        #             )
                
        #         # Use Pearson correlation as the primary metric for STS tasks
        #         model_list.append({"path": save_dir_path, "score": eval_ndcg})
        #         model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)  # Higher is better
                
        #         if len(model_list) > args.keep_best_n_checkpoints:
        #             removed_model = model_list.pop(-1)  # Remove worst model
        #             shutil.rmtree(removed_model["path"])

        #         log_rank(f"Model has been saved to {save_dir_path}")
        
        # dist.barrier()

        # log_rank(f"Finally Evaluating before saving model... epoch {epoch} step {logging_output['global_step']}")
        # eval_mrr, eval_ndcg, eval_map = evaluate(args, tokenizer, model.module.student_model, dataset["dev"], "dev", device)
        # if "test" in dataset: #evaluate for test, no affect
        #     _, _, _ = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)
        # ckpt_name = f"epoch{epoch + 1}_step{logging_output['global_step']}_mrr_{eval_mrr:.4f}_map{eval_map:.4f}_ndcg{eval_ndcg:.4f}"
        ckpt_name = f"epoch{epoch + 1}_step{logging_output['global_step']}"
        save_dir_path = os.path.join(args.save_dir, ckpt_name)
        
        os.makedirs(save_dir_path, exist_ok=True)
        if not args.only_save_projector:
            log_rank("Saving tokenizer...")
            tokenizer.save_pretrained(save_dir_path)
            log_rank("Saving model...")
            model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
            log_rank("Saving config")
            model.module.student_model.config.save_pretrained(save_dir_path)
        
        if hasattr(model.module, "projectors"):
            log_rank("Saving projector...")
            torch.save(
                model.module.projectors.state_dict(), 
                os.path.join(save_dir_path, "projector.pt")
            )
        
        # Use Pearson correlation as the primary metric for STS tasks
        model_list.append({"path": save_dir_path, "score": eval_ndcg})
        model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)  # Higher is better
        
        if len(model_list) > args.keep_best_n_checkpoints:
            removed_model = model_list.pop(-1)  # Remove worst model
            shutil.rmtree(removed_model["path"])

        log_rank(f"Model has been saved to {save_dir_path}")
        # eval_mrr, eval_ndcg, eval_map = evaluate(args, tokenizer, model.module.student_model, dataset["test"], "test", device)

    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad()
def evaluate(args, tokenizer, student_model, dataset, split, device):
    batch_size=64
    if dist.get_rank() != 0:
        return None, None, None        
    
    student_model.eval()

    corpus = dataset.corpus
    queries = dataset.queries
    qrels = dataset.qrels

    # Encode corpus
    doc_ids = list(corpus.keys())
    print('# document in corpus', len(doc_ids))
    doc_texts = [v['title'] + ' ' + v['text'] for v in list(corpus.values())]

    doc_embeddings = []
    # for i in range(0, len(doc_texts), batch_size):
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding corpus"):
        batch = doc_texts[i:i+batch_size]

        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length).to(student_model.device)

        outputs = student_model(**inputs)  # BERTModel or LLM2vec model
        if args.peft: #LLM2vec
            token_embeddings = outputs.last_hidden_state           # [B, T, D]
            attention_mask = inputs['attention_mask']              # [B, T]
            # Expand mask: [B, T] → [B, T, 1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

            # Masked sum then divide by actual token count
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)        # [B, D]
            sum_mask = input_mask_expanded.sum(dim=1)                                         # [B, 1]
            emb = sum_embeddings / sum_mask                                           # [B, D]
        else: #BERT 
            emb = outputs.last_hidden_state[:, 0] # [B, D] lay [cls] token

        doc_embeddings.append(emb)

    doc_embeddings = torch.cat(doc_embeddings, dim=0)  # (D, dim)
    print(f'doc_embedig shape {doc_embeddings.shape}')

    # Encode queries
    query_ids = list(queries.keys())
    queries_texts = list(queries.values())
    print('# query', len(query_ids))

    query_embeddings = []
    # for i in range(0, len(queries_texts), batch_size):
    for i in tqdm(range(0, len(queries_texts), batch_size), desc="Encoding queries"):

        batch = queries_texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length).to(student_model.device)

        outputs = student_model(**inputs)  # BERTModel outputs
        if args.peft: #LLM2vec
            token_embeddings = outputs.last_hidden_state           # [B, T, D]
            attention_mask = inputs['attention_mask']              # [B, T]
            # Expand mask: [B, T] → [B, T, 1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

            # Masked sum then divide by actual token count
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)        # [B, D]
            sum_mask = input_mask_expanded.sum(dim=1)                                         # [B, 1]
            emb = sum_embeddings / sum_mask                                           # [B, D]
        else:
            emb = outputs.last_hidden_state[:, 0] # [B, D] lay [cls] token
        query_embeddings.append(emb)
    
    query_embeddings = torch.cat(query_embeddings, dim=0)  # (Q, dim)
    print(f'query_embeddings shape {query_embeddings.shape}')

    # Compute similarity
    sim_matrix = cos_sim(query_embeddings, doc_embeddings)  # (Q, D)
    print(f'sim_matrix {sim_matrix.shape}')

    mrr_10, ndcg_10, map_100 = [], [], []

    # sim_matrix = sim_matrix.cpu().numpy()
    sim_matrix = sim_matrix.to(dtype=torch.float32).cpu().numpy()
    doc_ids_array = np.array(doc_ids)
    query_ids_array = np.array(query_ids)

    for q_idx, qid in enumerate(query_ids):
        relevant_docs = set(qrels.get(qid, {}).keys())  # tập tài liệu đúng
        if len(relevant_docs) == 0:
            continue

        # import pdb; pdb.set_trace()

        sim_scores = sim_matrix[q_idx]
        ranked_indices = np.argsort(-sim_scores)  # sắp xếp giảm dần
        ranked_doc_ids = doc_ids_array[ranked_indices]

        # === MRR@10 ===
        for rank, doc_id in enumerate(ranked_doc_ids[:10]):
            if doc_id in relevant_docs:
                mrr_10.append(1 / (rank + 1))
                break
        else:
            mrr_10.append(0.0)

        # === NDCG@10 ===
        dcg = 0.0
        for i, doc_id in enumerate(ranked_doc_ids[:10]):
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_docs), 10))])
        ndcg_10.append(dcg / idcg if idcg > 0 else 0.0)

        # === MAP@100 ===
        hits, precisions = 0, []
        for i, doc_id in enumerate(ranked_doc_ids[:100]):
            if doc_id in relevant_docs:
                hits += 1
                precisions.append(hits / (i + 1))
        map_100.append(np.mean(precisions) if precisions else 0.0)

    eval_info = {
        "MRR@10": np.mean(mrr_10),
        "NDCG@10": np.mean(ndcg_10),
        "MAP@100": np.mean(map_100),
    }


    print(f'eval_info - {split} - {eval_info}')

    student_model.train()

    return eval_info["MRR@10"], eval_info["NDCG@10"], eval_info["MAP@100"]

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # save arguments
    if dist.get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    print('user ds_config', ds_config)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    # import pdb; pdb.set_trace()

    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer_grouped_parameters,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
       
    if args.do_eval:
        evaluate(args, distiller.student_tokenizer, model_engine.module.student_model, dataset["test"], "test", device)
        
    
if __name__ == "__main__":
    main()
