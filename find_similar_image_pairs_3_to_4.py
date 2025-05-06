from pathlib import Path
import json
from tqdm import tqdm
import torch

from image_embedding_model import CLIPImageEncoder

transaction = 3
cat = 10
transaction3_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
transaction2_path = Path(f"multi-turn/transaction{transaction - 1}/transaction{transaction - 1}_{cat}.jsonl")
transaction1_path = Path(f"multi-turn/transaction{transaction - 2}/transaction{transaction - 2}_{cat}.jsonl")

caption_path = Path(f"captions/image_captions_{cat}.jsonl")

image_embedding_model = CLIPImageEncoder()

with open(transaction3_path, 'r', encoding='utf-8') as f:
    transactions3 = [json.loads(line) for line in f]

with open(transaction2_path, 'r', encoding='utf-8') as f:
    transactions2 = [json.loads(line) for line in f]

with open(transaction1_path, 'r', encoding='utf-8') as f:
    transactions1 = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

captions_total_image_embeds = []
captions_total_img_ids = []

batch_size = 16

for i in tqdm(range(0, len(captions), batch_size), desc="Batch Caption Images Embedding"):
    captions_batch_img_ids = [torch.tensor(int(img_caption["query_img_id"])) for img_caption in
                              captions[i:i + batch_size]]
    captions_batch_img_paths = [list(Path(f"data/{cat}").rglob(f"{query_img_id}.jpg"))[0] for query_img_id in
                                captions_batch_img_ids]
    captions_processed_image_batch = image_embedding_model.preprocess_images(captions_batch_img_paths)
    captions_batch_image_embeds = image_embedding_model.embed_images(captions_processed_image_batch)
    captions_total_image_embeds.extend(captions_batch_image_embeds)
    captions_total_img_ids.extend(captions_batch_img_ids)

captions_total_image_embeds = torch.stack(captions_total_image_embeds)
captions_total_img_ids = torch.stack(captions_total_img_ids)

t3_total_image_embeds = []
t2_total_image_embeds = []
t1_total_image_embeds = []
t0_total_image_embeds = []

for i in tqdm(range(0, len(transactions3), batch_size), desc="Batch Transaction Images Embedding"):
    batch_t3_img_ids = [torch.tensor(int(img["query_img_id"])) for img in
                        transactions3[i:i + batch_size]]
    batch_t3_img_paths = [list(Path(f"data/{cat}").rglob(f"{t3_img_id}.jpg"))[0] for t3_img_id in
                          batch_t3_img_ids]
    batch_t3_processed_images = image_embedding_model.preprocess_images(batch_t3_img_paths)
    batch_t3_img_embeds = image_embedding_model.embed_images(batch_t3_processed_images)

    batch_t2_img_ids = [torch.tensor(int(img["after_img_id"])) for img in
                        transactions1[i:i + batch_size]]
    batch_t2_img_paths = [list(Path(f"data/{cat}").rglob(f"{t2_img_id}.jpg"))[0] for t2_img_id in
                          batch_t2_img_ids]
    batch_t2_processed_images = image_embedding_model.preprocess_images(batch_t2_img_paths)
    batch_t2_img_embeds = image_embedding_model.embed_images(batch_t2_processed_images)

    batch_t1_img_ids = [torch.tensor(int(img["query_img_id"])) for img in
                        transactions1[i:i + batch_size]]
    batch_t1_img_paths = [list(Path(f"data/{cat}").rglob(f"{t1_img_id}.jpg"))[0] for t1_img_id in
                          batch_t1_img_ids]
    batch_t1_processed_images = image_embedding_model.preprocess_images(batch_t1_img_paths)
    batch_t1_img_embeds = image_embedding_model.embed_images(batch_t1_processed_images)

    batch_t0_img_ids = [torch.tensor(int(img["before_img_id"])) for img in
                        transactions1[i:i + batch_size]]
    batch_t0_img_paths = [list(Path(f"data/{cat}").rglob(f"{t0_img_id}.jpg"))[0] for t0_img_id in
                          batch_t0_img_ids]
    batch_t0_processed_images = image_embedding_model.preprocess_images(batch_t0_img_paths)
    batch_t0_img_embeds = image_embedding_model.embed_images(batch_t0_processed_images)

    t3_total_image_embeds.extend(batch_t3_img_embeds)
    t2_total_image_embeds.extend(batch_t2_img_embeds)
    t1_total_image_embeds.extend(batch_t1_img_embeds)
    t0_total_image_embeds.extend(batch_t0_img_embeds)

t3_total_image_embeds = torch.stack(t3_total_image_embeds)
t2_total_image_embeds = torch.stack(t2_total_image_embeds)
t1_total_image_embeds = torch.stack(t1_total_image_embeds)
t0_total_image_embeds = torch.stack(t0_total_image_embeds)

# combination true matrix
cosine_similarity_matrix_with_t0 = torch.matmul(t0_total_image_embeds, captions_total_image_embeds.T)
mask_t0 = (cosine_similarity_matrix_with_t0 > 0.92) & (cosine_similarity_matrix_with_t0 < 0.95)

cosine_similarity_matrix_with_t1 = torch.matmul(t1_total_image_embeds, captions_total_image_embeds.T)
mask_t1 = cosine_similarity_matrix_with_t1 < 0.90

cosine_similarity_matrix_with_t2 = torch.matmul(t2_total_image_embeds, captions_total_image_embeds.T)
mask_t2 = cosine_similarity_matrix_with_t2 < 0.90

cosine_similarity_matrix_with_t3 = torch.matmul(t3_total_image_embeds, captions_total_image_embeds.T)
mask_t3 = cosine_similarity_matrix_with_t3 < 0.90

filtered_cosine_similarity_matrix_if_rollback_combination = cosine_similarity_matrix_with_t0.clone()
filtered_cosine_similarity_matrix_if_rollback_combination[~(mask_t0 & mask_t1 & mask_t2 & mask_t3)] = 0
topk_values_if_rollback_combination, topk_indices_if_rollback_combination = torch.topk(
    filtered_cosine_similarity_matrix_if_rollback_combination, k=5, dim=1,
    largest=True)
topk_ids_if_rollback_combination = captions_total_img_ids[topk_indices_if_rollback_combination]

# combination false matrix
mask_t0 = cosine_similarity_matrix_with_t0 < 0.90
mask_t1 = (cosine_similarity_matrix_with_t1 > 0.92) & (cosine_similarity_matrix_with_t1 < 0.95)

filtered_cosine_similarity_matrix_if_combination_not = cosine_similarity_matrix_with_t1.clone()
filtered_cosine_similarity_matrix_if_combination_not[~(mask_t0 & mask_t1 & mask_t2 & mask_t3)] = 0
topk_values_if_combination_not, topk_indices_if_combination_not = torch.topk(
    filtered_cosine_similarity_matrix_if_combination_not, k=5, dim=1,
    largest=True)
topk_ids_if_combination_not = captions_total_img_ids[topk_indices_if_combination_not]

# rollback false matrix
mask_t0 = cosine_similarity_matrix_with_t0 < 0.90
mask_t1 = cosine_similarity_matrix_with_t1 < 0.90
mask_t2 = cosine_similarity_matrix_with_t2 < 0.90
mask_t3 = cosine_similarity_matrix_with_t3 < 0.95

filtered_cosine_similarity_matrix_if_rollback_not = cosine_similarity_matrix_with_t3.clone()
filtered_cosine_similarity_matrix_if_rollback_not[~(mask_t0 & mask_t1 & mask_t2 & mask_t3)] = 0
topk_values_if_rollback_not, topk_indices_if_rollback_not = torch.topk(
    filtered_cosine_similarity_matrix_if_rollback_not, k=5, dim=1,
    largest=True)
topk_ids_if_rollback_not = captions_total_img_ids[topk_indices_if_rollback_not]

for i, t3 in enumerate(tqdm(transactions3, desc="Find Similar Images")):
    combi_best_score = topk_values_if_rollback_combination[i][0]
    combi_not_best_score = topk_values_if_combination_not[i][0]

    has_rollback = transactions1[i]["rollback"]
    has_combination = transactions2[i]["combination"]

    if has_rollback and has_combination and combi_best_score != 0:
        t3["after_img_id"] = str(topk_ids_if_rollback_combination[i][0].item())
        t3["similarity_score"] = round(topk_values_if_rollback_combination[i][0].item(), 3)
        t3["rollback"] = 3

        t4_captions = \
            [c["query_captions"] for c in captions if
             c["query_img_id"] == str(topk_ids_if_rollback_combination[i][0].item())][0]
        transaction4 = {"id": t3["id"],
                        "query_img_id": str(topk_ids_if_rollback_combination[i][0].item()),
                        "query_captions": t4_captions,
                        "before_img_id": t3["query_img_id"],
                        "after_img_id": None,
                        "similarity_score": None,
                        "combination": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }
    elif has_rollback and not has_combination and combi_not_best_score != 0:
        t3["after_img_id"] = str(topk_ids_if_combination_not[i][0].item())
        t3["similarity_score"] = round(topk_values_if_combination_not[i][0].item(), 3)
        t3["rollback"] = 2

        t4_captions = \
            [c["query_captions"] for c in captions if
             c["query_img_id"] == str(topk_ids_if_combination_not[i][0].item())][0]
        transaction4 = {"id": t3["id"],
                        "query_img_id": str(topk_ids_if_combination_not[i][0].item()),
                        "query_captions": t4_captions,
                        "before_img_id": t3["query_img_id"],
                        "after_img_id": None,
                        "similarity_score": None,
                        "combination": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }
    elif not has_rollback and not has_combination and combi_not_best_score != 0:
        t3["after_img_id"] = str(topk_ids_if_combination_not[i][0].item())
        t3["similarity_score"] = round(topk_values_if_combination_not[i][0].item(), 3)
        t3["rollback"] = 1

        t4_captions = \
            [c["query_captions"] for c in captions if
             c["query_img_id"] == str(topk_ids_if_combination_not[i][0].item())][0]
        transaction4 = {"id": t3["id"],
                        "query_img_id": str(topk_ids_if_combination_not[i][0].item()),
                        "query_captions": t4_captions,
                        "before_img_id": t3["query_img_id"],
                        "after_img_id": None,
                        "similarity_score": None,
                        "combination": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }
    else:
        t3["after_img_id"] = str(topk_ids_if_rollback_not[i][0].item())
        t3["similarity_score"] = round(topk_values_if_rollback_not[i][0].item(), 3)
        t3["rollback"] = 0

        t4_captions = \
            [c["query_captions"] for c in captions if
             c["query_img_id"] == str(topk_ids_if_rollback_not[i][0].item())][0]
        transaction4 = {"id": t3["id"],
                        "query_img_id": str(topk_ids_if_rollback_not[i][0].item()),
                        "query_captions": t4_captions,
                        "before_img_id": t3["query_img_id"],
                        "after_img_id": None,
                        "similarity_score": None,
                        "combination": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }

    with open(f"multi-turn/transaction{transaction}_tmp/transaction{transaction}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(t3, ensure_ascii=False)
        f.write(json_line + "\n")
    with open(f"multi-turn/transaction{transaction + 1}_tmp/transaction{transaction + 1}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(transaction4, ensure_ascii=False)
        f.write(json_line + "\n")
