from typing import Set, Optional, Union, Any, Iterable, Tuple

from data.pair_dataset import PairDataset


class PairSet:
    def __init__(self, existing_set: Optional[Union[Set, 'PairSet', PairDataset]] = None):
        if isinstance(existing_set, PairSet):
            self.pair_set = existing_set.pair_set
        elif isinstance(existing_set, PairDataset):
            self.pair_set = set([tuple(input_example.texts) for input_example in existing_set.input_examples])
        else:
            self.pair_set = existing_set if existing_set is not None else set()

    def __contains__(self, item: Tuple[Any, Any]) -> bool:
        return item in self.pair_set

    def __iter__(self) -> Iterable[Tuple[Any, Any]]:
        return iter(self.pair_set)

    def add(self, key: Tuple[Any, Any]) -> bool:
        key_a, key_b = key
        if (key_a, key_b) in self.pair_set or (key_b, key_a) in self.pair_set:
            return False

        self.pair_set.add(key)
        return True

    def update(self, other: Set[Tuple[Any, Any]]):
        self.pair_set.update(other)
