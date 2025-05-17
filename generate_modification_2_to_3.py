from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm

transaction = 2
cat = 196

transaction_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
caption_path = Path(f"captions/image_captions_{cat}.jsonl")

with open(transaction_path, 'r', encoding='utf-8') as f:
    transactions = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

client = OpenAI(api_key="sk-proj"
                        "-qG08NSmx0cuc6MvV450NjGnJk52BxVWQITRt5cfeWueIGkv18saBzn0fj4vPTaBW0_XlGlsg_AT3BlbkFJLcctdvc1RO1"
                        "nSaPkqr7GbJf9qY4YfJ1A4KdpYZE955sV6bRGVS9AXIwUpCn8RySulrhWIjUkQA")


def generate_modification(query1, query2, target, combination=False):
    try:
        if combination == 2:
            prompt = (
                f"Transaction1 Image: '{query1}'\nTransaction2 Image: '{query2}'\nTarget Image: '{target}'\n"
                "Precondition:\n"
                "- Infer the shared fashion attributes between the transaction1 and the target "
                "and refer to it as 'Commonality 1'.\n"
                "- At the same time, infer the shared fashion attributes between the transaction2 and the target "
                "and refer to it as 'Commonality 2'.\n"
                "- Infer the unique fashion characteristics of the target that are not revealed in the transaction1"
                " and the transaction2 and refer to it as 'Adjustment'.\n"
                "Question: Describe the modification that transforms the image into the target by maintaining the "
                "attributes from 'Commonality 1' and 'Commonality 2', while applying the changes specified in "
                "'Adjustment'.\n"
                "Instruction:\n"
                "- Focus only on the garments described in each image, and infer the required fashion attributes "
                "(color, fabric, silhouette, details, pattern, length, neckline, sleeves, collar, closure, "
                "embellishment, construction, layering, etc.) based on distinct visual features.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'For making adjustments using the two preceding images as "
                "a reference, '."
            )
        elif combination == 1:
            prompt = (
                f"Initial Image: '{query1}'\nRecent Image: '{query2}'\nTarget Image: '{target}'\n"
                "Precondition:\n"
                "- Infer the shared fashion attributes between the initial and the target "
                "and refer to it as 'Commonality 1'.\n"
                "- At the same time, infer the shared fashion attributes between the recent and the target "
                "and refer to it as 'Commonality 2'.\n"
                "- Infer the unique fashion characteristics of the target that are not revealed in the initial"
                " and the recent and refer to it as 'Adjustment'.\n"
                "Question: Describe the modification that transforms the image into the target by maintaining the "
                "attributes from 'Commonality 1' and 'Commonality 2', while applying the changes specified in "
                "'Adjustment'.\n"
                "Instruction:\n"
                "- Focus only on the garments described in each image, and infer the required fashion attributes "
                "(color, fabric, silhouette, details, pattern, length, neckline, sleeves, collar, closure, "
                "embellishment, construction, layering, etc.) based on distinct visual features.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'In order to make changes based on the most recent and the very "
                "first image as a reference, '."
            )
        else:
            prompt = (
                f"Latest Image: '{query2}'\nTarget Image: '{target}'\n"
                "Question: Describe the modifications to transform the latest into the target in a sentence.\n"
                "Instruction:\n"
                "- Focus only on the garments described in the latest and the target, and generate modifications "
                "based on their distinct visual differences.\n"
                "- You only need to describe the most distinct difference.\n"
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


for idx, t in tqdm(enumerate(transactions), total=len(transactions), desc="Generating modifications"):
    next_transaction_id = t["after_img_id"]
    next_captions = [c["query_captions"] for c in captions if c["query_img_id"] == next_transaction_id][0]

    is_combination = t["combination"]
    if is_combination == 2:
        before_img_id = t["before_img_id"]
        before_captions = [c["query_captions"] for c in captions if c["query_img_id"] == before_img_id][0]
        query_img_id = t["query_img_id"]
        query_captions = [c["query_captions"] for c in captions if c["query_img_id"] == query_img_id][0]
        modification = generate_modification(before_captions, query_captions, next_captions, is_combination)
        t["modifier"] = modification
    elif is_combination == 1:
        initial_img_id = t["id"]
        initial_captions = [c["query_captions"] for c in captions if c["query_img_id"] == initial_img_id][0]
        query_img_id = t["query_img_id"]
        query_captions = [c["query_captions"] for c in captions if c["query_img_id"] == query_img_id][0]
        modification = generate_modification(initial_captions, query_captions, next_captions, is_combination)
        t["modifier"] = modification
    else:  # combination not occur
        query_img_id = t["query_img_id"]
        query_captions = t["query_captions"]
        modification = generate_modification(None, query_captions, next_captions, is_combination)
        t["modifier"] = modification

    with open(f"multi-turn/transaction{transaction}_tmp/transaction{transaction}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(t, ensure_ascii=False)
        f.write(json_line + "\n")
