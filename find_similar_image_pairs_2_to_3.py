from pathlib import Path
import json
from tqdm import tqdm
import torch

from image_embedding_model import CLIPImageEncoder

transaction = 1
cat = 196
transaction_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
caption_path = Path(f"captions/image_captions_{cat}.jsonl")

image_embedding_model = CLIPImageEncoder()

with open(transaction_path, 'r', encoding='utf-8') as f:
    transactions = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

captions_total_image_embeds = []
captions_total_img_ids = []

batch_size = 16

for i in tqdm(range(0, len(captions), batch_size), desc="Batch Caption Images Embedding"):
    captions_batch_query_img_ids = [torch.tensor(int(img_caption["query_img_id"])) for img_caption in
                                    captions[i:i + batch_size]]
    captions_batch_query_img_paths = [list(Path(f"data/{cat}").rglob(f"{query_img_id}.jpg"))[0] for query_img_id in
                                      captions_batch_query_img_ids]
    captions_processed_image_batch = image_embedding_model.preprocess_images(captions_batch_query_img_paths)
    captions_batch_image_embeds = image_embedding_model.embed_images(captions_processed_image_batch)
    captions_total_image_embeds.extend(captions_batch_image_embeds)
    captions_total_img_ids.extend(captions_batch_query_img_ids)

captions_total_image_embeds = torch.stack(captions_total_image_embeds)
captions_total_img_ids = torch.stack(captions_total_img_ids)

query_total_image_embeds = []
before_total_image_embeds = []
after_total_image_embeds = []

for i in tqdm(range(0, len(transactions), batch_size), desc="Batch Transaction Images Embedding"):
    batch_query_img_ids = [torch.tensor(int(img["query_img_id"])) for img in
                           transactions[i:i + batch_size]]  # transaction1
    batch_query_img_paths = [list(Path(f"data/{cat}").rglob(f"{query_img_id}.jpg"))[0] for query_img_id in
                             batch_query_img_ids]
    batch_query_processed_images = image_embedding_model.preprocess_images(batch_query_img_paths)
    batch_query_img_embeds = image_embedding_model.embed_images(batch_query_processed_images)

    batch_before_img_ids = [torch.tensor(int(img["before_img_id"])) for img in
                            transactions[i:i + batch_size]]  # transaction0
    batch_before_img_paths = [list(Path(f"data/{cat}").rglob(f"{before_img_id}.jpg"))[0] for before_img_id in
                              batch_before_img_ids]
    batch_before_processed_images = image_embedding_model.preprocess_images(batch_before_img_paths)
    batch_before_img_embeds = image_embedding_model.embed_images(batch_before_processed_images)

    batch_after_img_ids = [torch.tensor(int(img["after_img_id"])) for img in
                           transactions[i:i + batch_size]]  # transaction2
    batch_after_img_paths = [list(Path(f"data/{cat}").rglob(f"{before_after_id}.jpg"))[0] for before_after_id in
                             batch_after_img_ids]
    batch_after_processed_images = image_embedding_model.preprocess_images(batch_after_img_paths)
    batch_after_img_embeds = image_embedding_model.embed_images(batch_after_processed_images)

    query_total_image_embeds.extend(batch_query_img_embeds)
    before_total_image_embeds.extend(batch_before_img_embeds)
    after_total_image_embeds.extend(batch_after_img_embeds)

query_total_image_embeds = torch.stack(query_total_image_embeds)
before_total_image_embeds = torch.stack(before_total_image_embeds)
after_total_image_embeds = torch.stack(after_total_image_embeds)

# rollback true matrix
cosine_similarity_matrix_with_query = torch.matmul(query_total_image_embeds, captions_total_image_embeds.T)
mask_query = (cosine_similarity_matrix_with_query > 0.92) & (cosine_similarity_matrix_with_query < 0.97)

cosine_similarity_matrix_with_after = torch.matmul(after_total_image_embeds, captions_total_image_embeds.T)
mask_after = (cosine_similarity_matrix_with_after > 0.92) & (cosine_similarity_matrix_with_after < 0.97)

cosine_similarity_matrix_with_before = torch.matmul(before_total_image_embeds, captions_total_image_embeds.T)
mask_before = cosine_similarity_matrix_with_before < 0.90

filtered_cosine_similarity_matrix1_if_rollback = cosine_similarity_matrix_with_query.clone()
filtered_cosine_similarity_matrix2_if_rollback = cosine_similarity_matrix_with_after.clone()
filtered_cosine_similarity_matrix1_if_rollback[~(mask_query & mask_after & mask_before)] = 0
filtered_cosine_similarity_matrix2_if_rollback[~(mask_query & mask_after & mask_before)] = 0
combi_topk_values1_if_rollback, combi_topk_indices1_if_rollback = torch.topk(
    filtered_cosine_similarity_matrix1_if_rollback, k=5, dim=1,
    largest=True)
combi_topk_values2_if_rollback, combi_topk_indices2_if_rollback = torch.topk(
    filtered_cosine_similarity_matrix2_if_rollback, k=5, dim=1,
    largest=True)
combi_topk_ids1_if_rollback = captions_total_img_ids[combi_topk_indices1_if_rollback]

# rollback false matrix
mask_query = cosine_similarity_matrix_with_query < 0.90
mask_after = (cosine_similarity_matrix_with_after > 0.92) & (cosine_similarity_matrix_with_after < 0.97)
mask_before = (cosine_similarity_matrix_with_before > 0.92) & (cosine_similarity_matrix_with_before < 0.97)

filtered_cosine_similarity_matrix1_if_rollback_not = cosine_similarity_matrix_with_before.clone()
filtered_cosine_similarity_matrix2_if_rollback_not = cosine_similarity_matrix_with_after.clone()
filtered_cosine_similarity_matrix1_if_rollback_not[~(mask_query & mask_after & mask_before)] = 0
filtered_cosine_similarity_matrix2_if_rollback_not[~(mask_query & mask_after & mask_before)] = 0
combi_topk_values1_if_rollback_not, combi_topk_indices1_if_rollback_not = torch.topk(
    filtered_cosine_similarity_matrix1_if_rollback_not, k=5, dim=1,
    largest=True)
combi_topk_values2_if_rollback_not, combi_topk_indices2_if_rollback_not = torch.topk(
    filtered_cosine_similarity_matrix2_if_rollback_not, k=5, dim=1,
    largest=True)
combi_topk_ids1_if_rollback_not = captions_total_img_ids[combi_topk_indices1_if_rollback_not]

# combination not matrix
mask_query = cosine_similarity_matrix_with_query < 0.90
mask_after = cosine_similarity_matrix_with_after < 0.95
mask_before = cosine_similarity_matrix_with_before < 0.90

filtered_cosine_similarity_matrix_if_combi_not = cosine_similarity_matrix_with_after.clone()
filtered_cosine_similarity_matrix_if_combi_not[~(mask_query & mask_after & mask_before)] = 0
topk_values, topk_indices = torch.topk(
    filtered_cosine_similarity_matrix_if_combi_not, k=5, dim=1,
    largest=True)
topk_ids = captions_total_img_ids[topk_indices]

for i, t in enumerate(tqdm(transactions, desc="Find Similar Images")):
    rollback_best_score = combi_topk_values1_if_rollback[i][0]
    rollback_not_best_score = combi_topk_values1_if_rollback_not[i][0]
    combi_not_best_score = topk_values[i][0]
    t2_captions = \
        [c["query_captions"] for c in captions if c["query_img_id"] == t["after_img_id"]][0]
    if (t["rollback"] and rollback_best_score == 0) or (not t["rollback"] and rollback_not_best_score == 0):
        if combi_not_best_score == 0:
            print(t["id"], "Error!")
        transaction2 = {"id": t["id"],
                        "query_img_id": t["after_img_id"],
                        "query_captions": t2_captions,
                        "before_img_id": t["query_img_id"],
                        "after_img_id": str(topk_ids[i][0].item()),
                        "similarity_score": round(topk_values[i][0].item(), 3),
                        "combination": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }
        t3_captions = \
            [c["query_captions"] for c in captions if c["query_img_id"] == str(topk_ids[i][0].item())][0]
        transaction3 = {"id": t["id"],
                        "query_img_id": str(topk_ids[i][0].item()),
                        "query_captions": t3_captions,
                        "before_img_id": t["after_img_id"],
                        "after_img_id": None,
                        "similarity_score": None,
                        "rollback": 0,
                        "modifier": None,
                        "gt_img_ids": None,
                        }
    else:  # combination occurs
        if t["rollback"]:
            average_best_scores = (combi_topk_values1_if_rollback[i][0].item() +
                                   combi_topk_values2_if_rollback[i][0].item()) / 2
            transaction2 = {"id": t["id"],
                            "query_img_id": t["after_img_id"],
                            "query_captions": t2_captions,
                            "before_img_id": t["query_img_id"],
                            "after_img_id": str(combi_topk_ids1_if_rollback[i][0].item()),
                            "similarity_score": round(average_best_scores, 3),
                            "combination": 2,  # rollback: True, combination: True
                            "modifier": None,
                            "gt_img_ids": None,
                            }
            t3_captions = \
                [c["query_captions"] for c in captions if
                 c["query_img_id"] == str(combi_topk_ids1_if_rollback[i][0].item())][0]
            transaction3 = {"id": t["id"],
                            "query_img_id": str(combi_topk_ids1_if_rollback[i][0].item()),
                            "query_captions": t3_captions,
                            "before_img_id": t["after_img_id"],
                            "after_img_id": None,
                            "similarity_score": None,
                            "rollback": 0,
                            "modifier": None,
                            "gt_img_ids": None,
                            }
        else:
            average_best_scores = (combi_topk_values1_if_rollback_not[i][0].item() +
                                   combi_topk_values2_if_rollback_not[i][0].item()) / 2
            transaction2 = {"id": t["id"],
                            "query_img_id": t["after_img_id"],
                            "query_captions": t2_captions,
                            "before_img_id": t["query_img_id"],
                            "after_img_id": str(combi_topk_ids1_if_rollback_not[i][0].item()),
                            "similarity_score": round(average_best_scores, 3),
                            "combination": 1,  # rollback: False, combination: True
                            "modifier": None,
                            "gt_img_ids": None,
                            }
            t3_captions = \
                [c["query_captions"] for c in captions if
                 c["query_img_id"] == str(combi_topk_ids1_if_rollback_not[i][0].item())][0]
            transaction3 = {"id": t["id"],
                            "query_img_id": str(combi_topk_ids1_if_rollback_not[i][0].item()),
                            "query_captions": t3_captions,
                            "before_img_id": t["after_img_id"],
                            "after_img_id": None,
                            "similarity_score": None,
                            "rollback": 0,
                            "modifier": None,
                            "gt_img_ids": None,
                            }
    with open(f"multi-turn/transaction{transaction + 1}_tmp/transaction{transaction + 1}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(transaction2, ensure_ascii=False)
        f.write(json_line + "\n")
    with open(f"multi-turn/transaction{transaction + 2}_tmp/transaction{transaction + 2}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(transaction3, ensure_ascii=False)
        f.write(json_line + "\n")
