""" recover_batch_results.py

Recovers item.summary values directly from OpenAI's batch records, for use
when local Batch.batches tracking has been lost (e.g. after a kernel
restart or accidental Batch.create() re-call).

Works because each output line's custom_id IS the item's global .id - we
don't need to know which local Batch object/filename produced it.

Usage:
    from recover_batch_results import recover_all_completed
    items, missing = recover_all_completed(items)  # items[i].id must equal i
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def recover_all_completed(items):
    """
    Walk every batch ever submitted on this account, and for any that
    completed, pull its output file and write summaries back onto `items`
    by matching custom_id (the original global item.id) - not by position.
    """
    id_lookup = {item.id: item for item in items if item.id is not None}

    after = None
    recovered = 0
    seen_batch_ids = set()
    completed_batches = 0

    while True:
        response = client.batches.list(limit=100, after=after)
        if not response.data:
            break

        for batch in response.data:
            if batch.id in seen_batch_ids:
                continue
            seen_batch_ids.add(batch.id)

            if batch.status == "completed" and batch.output_file_id:
                completed_batches += 1
                content = client.files.content(batch.output_file_id)
                text = content.text if hasattr(content, "text") else content.content.decode("utf-8")

                for line in text.splitlines():
                    if not line.strip():
                        continue
                    json_line = json.loads(line)
                    item_id = int(json_line["custom_id"])
                    summary = json_line["response"]["body"]["choices"][0]["message"]["content"]

                    if item_id in id_lookup:
                        # Guard against cross-contamination: this OpenAI account has
                        # other batch jobs (e.g. the Week 6 pricer exercise) using the
                        # same custom_id numbering scheme. Only accept output that
                        # actually matches our football SYSTEM_PROMPT's format.
                        if summary and summary.strip().startswith("Name:"):
                            id_lookup[item_id].summary = summary
                            recovered += 1

        if not getattr(response, "has_more", False):
            break
        after = response.data[-1].id

    print(f"Scanned {len(seen_batch_ids)} batches, {completed_batches} completed")
    print(f"Recovered {recovered} summaries")

    missing = [item for item in items if item.summary is None]
    print(f"{len(missing)} items still need processing")

    return items, missing
