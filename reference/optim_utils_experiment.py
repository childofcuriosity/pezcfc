# optim_utils copy 4.py — Codebook std 过滤（限制投影目标）

# - 新增 _get_filtered_codebook()：在 nn_project（最近邻投影）时，过滤掉 std 过小的 token（低 std = 接近零向量，无实际语义）
# - 加了模块级缓存 _FILTERED_CODEBOOK_CACHE 避免每步重复计算
# - 核心想法：只允许 embedding 投影到"有语义"的 token 上
import random
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping

import open_clip

import torch

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)

# 模块级缓存，避免每次 step 都重新过滤码本
_FILTERED_CODEBOOK_CACHE = {}


def _get_filtered_codebook(embedding_layer, std_threshold):
    """按 std >= std_threshold 精简码本，返回 (已归一化的筛后矩阵, 原始词表索引映射)。"""
    cache_key = (id(embedding_layer), float(std_threshold))
    if cache_key in _FILTERED_CODEBOOK_CACHE:
        return _FILTERED_CODEBOOK_CACHE[cache_key]

    with torch.no_grad():
        W = embedding_layer.weight                 # [V, D]
        stds = W.std(dim=-1)                       # [V]
        valid_indices = torch.nonzero(stds >= std_threshold, as_tuple=True)[0]

        if valid_indices.numel() == 0:
            raise ValueError(
                f"std_threshold={std_threshold} 过高，码本中无任何 token 满足；"
                f"当前码本 std 最大值为 {stds.max().item():.4f}"
            )

        filtered_normalized = normalize_embeddings(W[valid_indices])

    print(f"[nn_project] codebook filtered by std>={std_threshold}: "
          f"{valid_indices.numel()}/{W.shape[0]} tokens kept "
          f"(std range kept: [{stds[valid_indices].min().item():.4f}, "
          f"{stds[valid_indices].max().item():.4f}])")

    _FILTERED_CODEBOOK_CACHE[cache_key] = (filtered_normalized, valid_indices)
    return filtered_normalized, valid_indices


def nn_project(curr_embeds, embedding_layer, print_hits=False, std_threshold=None):
    with torch.no_grad():
        bsz, seq_len, emb_dim = curr_embeds.shape

        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds)   # queries

        if std_threshold is not None and std_threshold > 0:
            search_matrix, idx_map = _get_filtered_codebook(embedding_layer, std_threshold)
        else:
            search_matrix = normalize_embeddings(embedding_layer.weight)
            idx_map = None

        hits = semantic_search(curr_embeds, search_matrix,
                               query_chunk_size=curr_embeds.shape[0],
                               top_k=1,
                               score_function=dot_score)

        if print_hits:
            print(f"mean hits:{mean(hit[0]['score'] for hit in hits)}")

        local_indices = torch.tensor(
            [hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device
        )

        # 筛后的 id 映射回原词表 id，保证 token_embedding/解码都正确
        nn_indices = idx_map[local_indices] if idx_map is not None else local_indices
        nn_indices = nn_indices.reshape((bsz, seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, args, device):
    prompt_len = args.prompt_len

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids


def optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device):
    opt_iters = args.iter
    lr = args.lr
    weight_decay = args.weight_decay
    print_step = args.print_step
    batch_size = args.batch_size
    print_new_best = getattr(args, 'print_new_best', False)
    std_threshold = getattr(args, 'std_threshold', None)   # 新增

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args.loss_weight
    best_text = ""

    for step in range(opt_iters):
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]
            
        universal_target_features = all_target_features
        
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False,
            std_threshold=std_threshold,           
        )

        # get cosine similarity score with all target features
        with torch.no_grad():
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, universal_target_features)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()
        
        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
        
        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
        
        logits_per_image, _ = model.forward_text_embedding(padded_embeds, dummy_ids, target_features)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        loss = loss * args.loss_weight
        
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        
        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]
        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            if not print_new_best:
                per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
            print(per_step_message)

        if best_sim * args.loss_weight < universal_cosim_score * args.loss_weight:
            best_sim = universal_cosim_score
            best_text = decoded_text
            if print_new_best:
                print(f"new best cosine sim: {best_sim}")
                print(f"new best prompt: {best_text}")


    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def optimize_prompt(model, preprocess, args, device, target_images=None, target_prompts=None):
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.get_tokenizer(args.clip_model)

    # get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

    # optimize prompt
    learned_prompt = optimize_prompt_loop(model, tokenizer, token_embedding, all_target_features, args, device)

    return learned_prompt
    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.concatenate(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.concatenate(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()