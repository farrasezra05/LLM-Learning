import os
import json
import pickle
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.notebook import tqdm

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"

BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise scouting profile of a football player. Respond only in 
this format. Do not include market value or transfer fees.
Name: Player's full name
Position: eg Defender
Club: Current club name
Profile: 1 sentence on age and nationality
Attributes: 1 sentence on physical profile and playing style (height, preferred foot)"""

class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end

        self.filename = f"{start}_{end}.jsonl"

        self.file_id = None
        self.batch_id = None
        self.output_file_id = None

        self.done = False

        folder = Path("lite") if lite else Path("full")

        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER

        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item):
        body = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": item.full,
                },
            ],
        }

        line = {
            "custom_id": f"football2_{item.id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        return json.dumps(line)

    def make_file(self):
        batch_file = self.batches / self.filename

        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start:self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches / self.filename

        with batch_file.open("rb") as f:
            response = client.files.create(
                file=f,
                purpose="batch",
            )

        self.file_id = response.id

    def submit_batch(self):
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )

        self.batch_id = response.id

    def is_ready(self):
        response = client.batches.retrieve(self.batch_id)

        if response.status == "completed":
            self.output_file_id = response.output_file_id
            return True

        if response.status in ("failed", "expired", "cancelled"):
            print(f"Batch {self.batch_id} finished with status '{response.status}'")

            if response.errors:
                print(response.errors)

        return False

    def fetch_output(self):
        output_file = self.output / self.filename

        content = client.files.content(self.output_file_id)

        with open(output_file, "wb") as f:
            f.write(content.read())

    def apply_output(self):
        output_file = self.output / self.filename

        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)

                item_id = int(json_line["custom_id"].split("_")[1])

                summary = (
                    json_line["response"]["body"]["choices"][0]["message"]["content"]
                )

                self.items[item_id].summary = summary

        self.done = True

    @classmethod
    def create(cls, items, lite):
        cls.batches = []

        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))

            batch = Batch(items, start, end, lite)

            cls.batches.append(batch)

        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()

        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if batch.done:
                continue

            if batch.is_ready():
                batch.fetch_output()
                batch.apply_output()

        finished = [batch for batch in cls.batches if batch.done]

        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items

        for batch in cls.batches:
            batch.items = None

        with state.open("wb") as f:
            pickle.dump(cls.batches, f)

        for batch in cls.batches:
            batch.items = items

        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)

        for batch in cls.batches:
            batch.items = items

        print(f"Loaded {len(cls.batches)} batches")
    
    @classmethod
    def run_failed(cls, limit=2):
        submitted = 0

        for batch in tqdm(cls.batches):
            if submitted >= limit:
                break

            if batch.done:
                continue

            if batch.batch_id is None:
                batch.make_file()
                batch.send_file()
                batch.submit_batch()

                submitted += 1
                print(f"Submitted {batch.filename}")
                continue

            response = client.batches.retrieve(batch.batch_id)
            status = response.status

            if status == "failed":
                print(f"Resubmitting {batch.filename}")

                batch.send_file()
                batch.submit_batch()

                submitted += 1

            else:
                print(f"Skipping {batch.filename} ({status})")

        print(f"\nSubmitted {submitted} batch(es).")
    
    @classmethod
    def run_until_done(cls, max_active=2, poll_interval=60):
        """
        Keep the OpenAI queue filled with up to max_active active batches.
        Automatically fetch completed batches and resubmit failed ones.
        """

        while True:

            # -----------------------------
            # Fetch completed batches
            # -----------------------------
            cls.fetch()

            active = 0
            remaining = 0

            for batch in cls.batches:

                if batch.done:
                    continue

                remaining += 1

                if batch.batch_id is None:
                    continue

                status = client.batches.retrieve(batch.batch_id).status

                if status in ("validating", "in_progress", "finalizing"):
                    active += 1

            print(f"\nRemaining: {remaining} | Active: {active}")

            if remaining == 0:
                print("🎉 All batches finished!")
                break

            # -----------------------------
            # Fill available slots
            # -----------------------------
            available = max_active - active

            if available > 0:
                cls.run_failed(limit=available)

            print(f"Sleeping {poll_interval} seconds...\n")
            time.sleep(poll_interval)