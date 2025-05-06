from pathlib import Path
import json
from tqdm import tqdm
import torch

from image_embedding_model import CLIPImageEncoder

captions_dir = Path("cir/captions")
captions_files = [file for file in captions_dir.glob("**/*.jsonl") if not file.is_dir()]

image_embedding_model = CLIPImageEncoder()

for caption_file in captions_files:
    cat1 = caption_file.stem.split("_")[-1]
    with open(caption_file, 'r', encoding='utf-8') as file:
        captions = [json.loads(line) for line in file]

    # 1. 검색 대상: embed images
    batch_size = 32

    captions_total_image_embeds = []
    captions_total_img_ids = []

    for i in tqdm(range(0, len(captions), batch_size), desc="Batch Images Embedding"):
        captions_batch_query_img_ids = [torch.tensor(int(img_caption["query_img_id"])) for img_caption in
                                        captions[i:i + batch_size]]
        captions_batch_query_img_paths = [list(Path(f"data/{cat1}").rglob(f"{query_img_id}.jpg"))[0] for query_img_id in
                                          captions_batch_query_img_ids]
        captions_processed_image_batch = image_embedding_model.preprocess_images(captions_batch_query_img_paths)
        captions_batch_image_embeds = image_embedding_model.embed_images(captions_processed_image_batch)
        captions_total_image_embeds.extend(captions_batch_image_embeds)
        captions_total_img_ids.extend(captions_batch_query_img_ids)

    captions_total_image_embeds = torch.stack(captions_total_image_embeds)
    captions_total_img_ids = torch.stack(captions_total_img_ids)

    # 2. calculate cosine similarity matrix
    cosine_similarity_matrix = torch.matmul(captions_total_image_embeds, captions_total_image_embeds.T)

    # 3. find similar image pairs
    mask = cosine_similarity_matrix <= 0.92
    filtered_cosine_similarity_matrix = cosine_similarity_matrix.clone()
    filtered_cosine_similarity_matrix[~mask] = 0
    topk_values, topk_indices = torch.topk(filtered_cosine_similarity_matrix, k=10, dim=1, largest=True)
    topk_img_ids = captions_total_img_ids[topk_indices]

    transaction0 = [{"id": c["query_img_id"],
                     "query_img_id": c["query_img_id"],
                     "query_captions": c["query_captions"],
                     "before_img_id": None,
                     "after_img_id": str(topk_img_ids[i][0].item()),
                     "similarity_score": round(topk_values[i][0].item(), 3),
                     "modifier": None,
                     "gt_img_ids": None
                     } for i, c in enumerate(captions)]

    with open(f"cir/multi-turn/transaction0/transaction0_{cat1}.jsonl", "a", encoding="utf-8") as f:
        for item in transaction0:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
