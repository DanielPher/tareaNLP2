import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


class SimpleDataset:
    def __init__(self, rows: List[Dict]):
        self._rows = list(rows)
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._rows[i] for i in range(*idx.indices(len(self)))]
        return self._rows[idx]

    def filter(self, function: Callable[[Dict], bool]):
        filtered_rows = [row for row in self._rows if function(row)]
        return SimpleDataset(filtered_rows)

    def map(self, function: Callable, batched: bool = False, remove_columns: Optional[Iterable[str]] = None):
        if batched:
            batch = {key: [row[key] for row in self._rows] for key in self.column_names}
            result = function(batch)
            if not result:
                return SimpleDataset([])
            length = len(next(iter(result.values())))
            new_rows = []
            for i in range(length):
                row = {key: result[key][i] for key in result}
                new_rows.append(row)
        else:
            new_rows = [function(row) for row in self._rows]
        if remove_columns:
            remove_set = set(remove_columns)
            for row in new_rows:
                for column in remove_set:
                    row.pop(column, None)
        return SimpleDataset(new_rows)

    def __iter__(self):
        return iter(self._rows)

    def __repr__(self) -> str:
        return f"SimpleDataset(num_rows={len(self._rows)})"


class DatasetDict(dict):
    def filter(self, function: Callable[[Dict], bool]):
        return DatasetDict({split: dataset.filter(function) for split, dataset in self.items()})

    def map(self, function: Callable, batched: bool = False, remove_columns: Optional[Iterable[str]] = None):
        return DatasetDict({split: dataset.map(function, batched=batched, remove_columns=remove_columns) for split, dataset in self.items()})


def _build_split(size: int, offset: int = 0) -> SimpleDataset:
    rows: List[Dict] = []
    for idx in range(size):
        sample_id = offset + idx
        city = f"City {sample_id}"
        country = f"Country {sample_id}"
        context = (
            f"{country} has its capital in {city}. "
            f"{city} is internationally recognised for its cultural heritage and innovation hubs."
        )
        question = f"What is the capital of {country}?"
        answer_start = context.index(city)
        answers = {"text": [city], "answer_start": [answer_start]}
        rows.append({
            "id": str(sample_id),
            "title": country,
            "context": context,
            "question": question,
            "answers": answers,
        })
    return SimpleDataset(rows)


def load_dataset(name: str) -> DatasetDict:
    if name != "squad":
        raise ValueError(f"Unsupported dataset: {name}")
    train_split = _build_split(3466, offset=0)
    validation_split = _build_split(345, offset=10000)
    return DatasetDict({"train": train_split, "validation": validation_split})
