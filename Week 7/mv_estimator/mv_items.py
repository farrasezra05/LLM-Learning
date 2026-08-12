""" This mv_items.py file is a modified version of the pricer_items.py file.
It is used to create a dataset of football players with their market values. """

from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self

PREFIX = "Market value is €"
QUESTION = "What is this football player's transfer market value to nearest euro?"

class PlayerItem(BaseModel):
    """
    A PlayerItem is a data-point of a football player with a market value.
    Mirrors the Item pattern from pricer_items.py: title/category/price -> name/position/value.
    """

    name: str
    position: str
    sub_position: Optional[str] = None
    value: float  # market_value_in_eur - the target, analogous to "price"
    foot: Optional[str] = None
    height_in_cm: Optional[float] = None
    country_of_citizenship: Optional[str] = None
    current_club_name: Optional[str] = None
    contract_expiration_date: Optional[str] = None
    date_of_birth: Optional[str] = None
    full: Optional[str] = None  # rendered text profile, analogous to product "full" text
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.value)}"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.name} = €{self.value:,.0f}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push PlayerItem lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load PlayerItem lists from HuggingFace Hub"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
        
    # Newly added functions in Week 7
    def count_tokens(self, tokenizer):
        """[NEW - WEEK 7] Count tokens in the summary"""
        return len(tokenizer.encode(self.summary, add_special_tokens=False))
    
    def make_prompts(self, tokenizer, max_tokens, do_round):
        """[NEW - WEEK 7] Make prompts and completions"""
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        if len(tokens) > max_tokens:
            summary = tokenizer.decode(tokens[:max_tokens]).rstrip()
        else:
            summary = self.summary
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.value)}" if do_round else str(self.value)

    def count_prompt_tokens(self, tokenizer):
        """[NEW - WEEK 7] Count tokens in the prompt"""
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    def to_datapoint(self) -> dict:
        """[NEW - WEEK 7]"""
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        """[NEW - WEEK 7] Push Item lists to HuggingFace Hub in prompt-completion format for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)
