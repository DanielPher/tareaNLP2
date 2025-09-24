from typing import Iterable, List, Sequence


_options = {}


def set_option(option: str, value):
    _options[option] = value


class Series:
    def __init__(self, data: Iterable):
        self._data = list(data)

    def sum(self):
        return sum(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        return str(self._data)


class DataFrame:
    def __init__(self, data, columns: Sequence[str] = None):
        if columns is None:
            if isinstance(data, dict):
                columns = list(data.keys())
            elif data:
                columns = [f"col_{i}" for i in range(len(data[0]))]
            else:
                columns = []
        self.columns = list(columns)

        if isinstance(data, dict):
            length = len(next(iter(data.values()))) if data else 0
            self._data = [[data[col][i] for col in self.columns] for i in range(length)]
        else:
            self._data = [list(row) for row in data]

    def __getitem__(self, key):
        if isinstance(key, list):
            indices = [self.columns.index(col) for col in key]
            rows = [[row[i] for i in indices] for row in self._data]
            return DataFrame(rows, columns=key)
        index = self.columns.index(key)
        return Series(row[index] for row in self._data)

    def sort_values(self, column: str, ascending: bool = True):
        idx = self.columns.index(column)
        sorted_rows = sorted(self._data, key=lambda row: row[idx], reverse=not ascending)
        return DataFrame(sorted_rows, columns=self.columns)

    def head(self, n: int = 5):
        return DataFrame(self._data[:n], columns=self.columns)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        lines: List[str] = [" | ".join(self.columns)]
        for row in self._data:
            lines.append(" | ".join(str(item) for item in row))
        return "\n".join(lines)


__all__ = ["DataFrame", "Series", "set_option"]
