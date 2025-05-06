from pathlib import Path
import json
from tqdm import tqdm
import torch

from image_embedding_model import CLIPImageEncoder

transaction = 1
cat = 66
transaction_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
caption_path = Path(f"captions/image_captions_{cat}.jsonl")

image_embedding_model = CLIPImageEncoder()

with open(transaction_path, 'r', encoding='utf-8') as f:
    transactions = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

captions_total_image_embeds = []
captions_total_img_ids = []

batch_size = 32

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

for i in tqdm(range(0, len(transactions), batch_size), desc="Batch Transaction Images Embedding"):
    batch_query_img_ids = [torch.tensor(int(img["query_img_id"])) for img in
                           transactions[i:i + batch_size]]
    batch_query_img_paths = [list(Path(f"data/{cat}").rglob(f"{query_img_id}.jpg"))[0] for query_img_id in
                             batch_query_img_ids]
    batch_query_processed_images = image_embedding_model.preprocess_images(batch_query_img_paths)
    batch_query_img_embeds = image_embedding_model.embed_images(batch_query_processed_images)
    batch_before_img_ids = [torch.tensor(int(img["before_img_id"])) for img in
                            transactions[i:i + batch_size]]
    batch_before_img_paths = [list(Path(f"data/{cat}").rglob(f"{before_img_id}.jpg"))[0] for before_img_id in
                              batch_before_img_ids]
    batch_before_processed_images = image_embedding_model.preprocess_images(batch_before_img_paths)
    batch_before_img_embeds = image_embedding_model.embed_images(batch_before_processed_images)

    query_total_image_embeds.extend(batch_query_img_embeds)
    before_total_image_embeds.extend(batch_before_img_embeds)

query_total_image_embeds = torch.stack(query_total_image_embeds)
before_total_image_embeds = torch.stack(before_total_image_embeds)

cosine_similarity_matrix_with_query = torch.matmul(query_total_image_embeds, captions_total_image_embeds.T)
mask_query = cosine_similarity_matrix_with_query < 0.90

cosine_similarity_matrix_with_before = torch.matmul(before_total_image_embeds, captions_total_image_embeds.T)
mask_before = cosine_similarity_matrix_with_before < 0.95

filtered_cosine_similarity_matrix_with_before = cosine_similarity_matrix_with_before.clone()
filtered_cosine_similarity_matrix_with_before[~(mask_query & mask_before)] = 0
before_topk_values, before_topk_indices = torch.topk(filtered_cosine_similarity_matrix_with_before, k=10, dim=1,
                                                     largest=True)
before_topk_ids = captions_total_img_ids[before_topk_indices]

mask_query = cosine_similarity_matrix_with_query < 0.92
mask_before = cosine_similarity_matrix_with_before < 0.88
filtered_cosine_similarity_matrix_with_query = cosine_similarity_matrix_with_query.clone()
filtered_cosine_similarity_matrix_with_query[~(mask_query & mask_before)] = 0
query_topk_values, query_topk_indices = torch.topk(filtered_cosine_similarity_matrix_with_query, k=10, dim=1,
                                                   largest=True)
query_topk_ids = captions_total_img_ids[query_topk_indices]

for i, t in enumerate(tqdm(transactions, desc="Find Similar Images")):
    if before_topk_values[i][0] > 0.92:  # rollback occurs
        current_transaction = {"id": transactions[i]["id"],
                               "query_img_id": transactions[i]["query_img_id"],
                               "query_captions": transactions[i]["query_captions"],
                               "before_img_id": transactions[i]["before_img_id"],
                               "after_img_id": str(before_topk_ids[i][0].item()),
                               "similarity_score": round(before_topk_values[i][0].item(), 3),
                               "rollback": 1,
                               "modifier": None,
                               "gt_img_ids": None,
                               }
        query_captions = \
            [c["query_captions"] for c in captions if c["query_img_id"] == str(before_topk_ids[i][0].item())][0]
        next_transaction = {"id": transactions[i]["id"],
                            "query_img_id": str(before_topk_ids[i][0].item()),
                            "query_captions": query_captions,
                            "before_img_id": transactions[i]["query_img_id"],
                            "after_img_id": None,
                            "similarity_score": None,
                            "combination": 0,
                            "modifier": None,
                            "gt_img_ids": None,
                            }
    else:
        current_transaction = {"id": transactions[i]["id"],
                               "query_img_id": transactions[i]["query_img_id"],
                               "query_captions": transactions[i]["query_captions"],
                               "before_img_id": transactions[i]["before_img_id"],
                               "after_img_id": str(query_topk_ids[i][0].item()),
                               "similarity_score": round(query_topk_values[i][0].item(), 3),
                               "rollback": 0,
                               "modifier": None,
                               "gt_img_ids": None,
                               }
        query_captions = \
            [c["query_captions"] for c in captions if c["query_img_id"] == str(query_topk_ids[i][0].item())][0]
        next_transaction = {"id": transactions[i]["id"],
                            "query_img_id": str(query_topk_ids[i][0].item()),
                            "query_captions": query_captions,
                            "before_img_id": transactions[i]["query_img_id"],
                            "after_img_id": None,
                            "similarity_score": None,
                            "combination": 0,
                            "modifier": None,
                            "gt_img_ids": None,
                            }
    with open(f"multi-turn/transaction{transaction}_tmp/transaction{transaction}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(current_transaction, ensure_ascii=False)
        f.write(json_line + "\n")
    with open(f"multi-turn/transaction{transaction + 1}_tmp/transaction{transaction + 1}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(next_transaction, ensure_ascii=False)
        f.write(json_line + "\n")
