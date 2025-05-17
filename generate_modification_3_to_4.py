from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm

transaction = 3
cat = 66

transaction3_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
transaction1_path = Path(f"multi-turn/transaction{transaction - 2}/transaction{transaction - 2}_{cat}.jsonl")
caption_path = Path(f"captions/image_captions_{cat}.jsonl")

with open(transaction3_path, 'r', encoding='utf-8') as f:
    transactions3 = [json.loads(line) for line in f]

with open(transaction1_path, 'r', encoding='utf-8') as f:
    transactions1 = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

client = OpenAI(api_key="sk-proj"
                        "-qG08NSmx0cuc6MvV450NjGnJk52BxVWQITRt5cfeWueIGkv18saBzn0fj4vPTaBW0_XlGlsg_AT3BlbkFJLcctdvc1RO1"
                        "nSaPkqr7GbJf9qY4YfJ1A4KdpYZE955sV6bRGVS9AXIwUpCn8RySulrhWIjUkQA")


def generate_modification(before, next, rollback=0):
    try:
        if rollback == 3:
            prompt = (
                f"Initial Image: '{before}'\nTarget Image: '{next}'\n"
                "Question: Describe the modifications to transform the initial into the target in a sentence.\n"
                "Instruction:\n"
                "- Focus only on the garments described in each image, and infer the fashion attributes "
                "(color, fabric, silhouette, details, pattern, length, neckline, sleeves, collar, closure, "
                "embellishment, construction, layering, etc.) based on distinct visual features.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'To modify the initial image so it aligns with the target version, "
                "'."
            )
        elif rollback == 2 or rollback == 1:
            prompt = (
                f"Transaction1 Image: '{before}'\nTarget Image: '{next}'\n"
                "Question: Describe the modifications to transform the transaction1 into the target in a sentence.\n"
                "Instruction:\n"
                "- Focus only on the garments described in each image, and infer the fashion attributes "
                "(color, fabric, silhouette, details, pattern, length, neckline, sleeves, collar, closure, "
                "embellishment, construction, layering, etc.) based on distinct visual features.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'To modify the transaction1 image so it aligns with the target "
                "version, '."
            )
        else:
            prompt = (
                f"Recent Image: '{before}'\nTarget Image: '{next}'\n"
                "Question: Describe the modifications to transform the recent into the target in a sentence.\n"
                "Instruction:\n"
                "- Focus only on the garments described in each image, and infer the fashion attributes "
                "(color, fabric, silhouette, details, pattern, length, neckline, sleeves, collar, closure, "
                "embellishment, construction, layering, etc.) based on distinct visual features.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'To convert the latest image to match the target appearance, '."

            )
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.output_text
    except Exception as e:
        print(f"[ERROR] {e}")
    return None


for idx, t in tqdm(enumerate(transactions3), total=len(transactions3), desc="Generating modifications"):
    next_transaction_id = t["after_img_id"]
    next_captions = [c["query_captions"] for c in captions if c["query_img_id"] == next_transaction_id][0]

    rollback_type = t["rollback"]
    if rollback_type == 3:
        before_img_id = t["id"]
        before_captions = [c["query_captions"] for c in captions if c["query_img_id"] == before_img_id][0]
        modification = generate_modification(before_captions, next_captions, rollback_type)
        t["modifier"] = "Compared to the most recent image, I prefer the initial image that I provided. " + modification
    elif rollback_type == 2 or rollback_type == 1:
        before_img_id = transactions1[idx]["query_img_id"]
        before_captions = [c["query_captions"] for c in captions if c["query_img_id"] == before_img_id][0]
        modification = generate_modification(before_captions, next_captions, rollback_type)
        t["modifier"] = "Compared to the most recent image, I prefer the transaction1 image. " + modification
    else:  # rollback not occur
        query_img_id = t["query_img_id"]
        query_captions = t["query_captions"]
        modification = generate_modification(query_captions, next_captions, rollback_type)
        t["modifier"] = modification

    with open(f"multi-turn/transaction{transaction}_tmp/transaction{transaction}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(t, ensure_ascii=False)
        f.write(json_line + "\n")
