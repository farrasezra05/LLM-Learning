""" mv_preprocess_sync.py

Fallback for when the OpenAI Batch API is degraded/stuck (as observed
2026-08-10 - fresh batches sitting at 0/1000 completed, cancellation itself
hanging - a known, recurring platform-side issue, not specific to this code).

Does the same job as mv_batch.py (LLM-rewrite each item's `full` text into a
clean `summary`), but via direct synchronous chat.completions.create() calls
run concurrently with a thread pool, so progress is visible in real time
instead of trusting an opaque batch queue.

Usage:
    from mv_preprocess_sync import rewrite_all
    rewrite_all(items, max_workers=20)
    # items[i].summary is now populated in place
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.notebook import tqdm

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3

SYSTEM_PROMPT = """Create a concise scouting profile of a football player. Respond only in this format. Do not include market value or transfer fees.
Name: Player's full name
Position: eg Defender
Club: Current club name
Profile: 1 sentence on age and nationality
Attributes: 1 sentence on physical profile and playing style (height, preferred foot)"""


def _rewrite_one(item):
    """Call the LLM for a single item, with basic retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item.full},
                ],
            )
            item.summary = response.choices[0].message.content
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed on {item.name} after {MAX_RETRIES} attempts: {e}")
                return False
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
    return False


def rewrite_all(items, max_workers=20, checkpoint_every=2000, checkpoint_fn=None):
    """
    Rewrite item.full -> item.summary for every item, concurrently.

    checkpoint_fn: optional callable(items, completed_count) - call this
    periodically (e.g. to pickle/save progress) so a long run isn't lost if
    interrupted. Not required, but recommended for 40K+ items.
    """
    failed = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(_rewrite_one, item): item for item in items}

        for future in tqdm(as_completed(future_to_item), total=len(items)):
            item = future_to_item[future]
            success = future.result()
            if not success:
                failed.append(item)
            completed += 1

            if checkpoint_fn and completed % checkpoint_every == 0:
                checkpoint_fn(items, completed)

    print(f"\nDone: {len(items) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        print("Retrying failed items once, sequentially...")
        still_failed = []
        for item in tqdm(failed):
            if not _rewrite_one(item):
                still_failed.append(item)
        print(f"After retry: {len(still_failed)} still failed")
        if still_failed:
            print("Names that never succeeded:", [it.name for it in still_failed[:20]])

    return items
