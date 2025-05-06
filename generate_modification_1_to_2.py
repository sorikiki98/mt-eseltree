from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm

transaction = 1
cat = 10

transaction_path = Path(f"multi-turn/transaction{transaction}/transaction{transaction}_{cat}.jsonl")
caption_path = Path(f"captions/image_captions_{cat}.jsonl")

with open(transaction_path, 'r', encoding='utf-8') as f:
    transactions = [json.loads(line) for line in f]

with open(caption_path, 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]

client = OpenAI(api_key="sk-proj"
                        "-qG08NSmx0cuc6MvV450NjGnJk52BxVWQITRt5cfeWueIGkv18saBzn0fj4vPTaBW0_XlGlsg_AT3BlbkFJLcctdvc1RO1"
                        "nSaPkqr7GbJf9qY4YfJ1A4KdpYZE955sV6bRGVS9AXIwUpCn8RySulrhWIjUkQA")


def generate_modification(current, next, rollback=False):
    try:
        if rollback:
            prompt = (
                f"Initial Image: '{current}'\nTarget Image: '{next}'\n"
                "Question: Describe the modifications to transform the initial into the target in a sentence.\n"
                "Instruction:\n"
                "- Focus only on the garments described in the initial and the target, and generate modifications "
                "based on their distinct visual differences.\n"
                "- You only need to describe the most distinct difference.\n"
                "- Do not use 'specify' or any synonyms of it.\n"
                "- Make a sentence that starts with 'To modify the initial image so it aligns with the target version, "
                "'."
            )
        else:
            prompt = (
                f"Latest Image: '{current}'\nTarget Image: '{next}'\n"
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
    is_rollback = t["rollback"]
    if is_rollback:
        before_img_id = t["before_img_id"]
        before_captions = [c["query_captions"] for c in captions if c["query_img_id"] == before_img_id][0]
        modification = generate_modification(before_captions, next_captions, is_rollback)
        t["modifier"] = "Compared to the most recent image, I prefer the initial image that I provided. " + modification
    else:
        before_img_id = t["query_img_id"]
        before_captions = t["query_captions"]
        modification = generate_modification(before_captions, next_captions, is_rollback)
        t["modifier"] = modification

    with open(f"multi-turn/transaction{transaction}_tmp/transaction{transaction}_{cat}.jsonl", "a",
              encoding="utf-8") as f:
        json_line = json.dumps(t, ensure_ascii=False)
        f.write(json_line + "\n")
