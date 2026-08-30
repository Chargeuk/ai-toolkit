import math
import random
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


DATASET_SAMPLING_STRATEGIES = (
    'combined',
    'round_robin',
    'weighted_round_robin',
)


def normalize_dataset_sampling_strategy(strategy: str) -> str:
    """Normalize legacy-friendly names while keeping combined shuffle as the default."""
    aliases = {
        'combined': 'combined',
        'shuffle': 'combined',
        'concat': 'combined',
        'round_robin': 'round_robin',
        'weighted_round_robin': 'weighted_round_robin',
    }
    try:
        return aliases[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset sampling strategy {strategy!r}. "
            f"Expected one of: {', '.join(DATASET_SAMPLING_STRATEGIES)}"
        ) from exc


def validate_sampling_weights(weights: Sequence[int], dataset_count: int) -> List[int]:
    if len(weights) != dataset_count:
        raise ValueError(f"Expected {dataset_count} sampling weights, got {len(weights)}")
    validated = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise ValueError(f"sampling_weight must be a positive integer, got {weight!r}")
        validated.append(weight)
    return validated


class DatasetRoundRobinSampler(Sampler[int]):
    """
    Interleave child datasets without duplicating them.

    Every child is shuffled independently. A child reshuffles and cycles when it
    is exhausted. The schedule length is rounded up to a complete weight cycle,
    so weights 3:1 produce exactly three slots from dataset 0 and one slot from
    dataset 1 in every four-slot cycle.
    """

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        weights: Sequence[int] | None = None,
        seed: int = 0,
    ):
        self.dataset_lengths = [int(length) for length in dataset_lengths]
        if not self.dataset_lengths:
            raise ValueError("Round-robin sampling requires at least one dataset")
        if any(length <= 0 for length in self.dataset_lengths):
            raise ValueError(f"Round-robin datasets must be non-empty, got lengths {self.dataset_lengths}")

        if weights is None:
            weights = [1] * len(self.dataset_lengths)
        self.weights = validate_sampling_weights(weights, len(self.dataset_lengths))
        self.seed = int(seed)
        self.epoch = 0

        self.offsets = []
        offset = 0
        for length in self.dataset_lengths:
            self.offsets.append(offset)
            offset += length

        weight_cycle = sum(self.weights)
        self.schedule_length = math.ceil(offset / weight_cycle) * weight_cycle

    def __len__(self) -> int:
        return self.schedule_length

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _shuffled_order(self, dataset_index: int, cycle: int, epoch: int) -> List[int]:
        order = list(range(self.dataset_lengths[dataset_index]))
        # Separate deterministic streams for every dataset, cycle, and epoch.
        local_seed = self.seed + epoch * 1_000_003 + dataset_index * 10_007 + cycle * 101
        random.Random(local_seed).shuffle(order)
        return order

    def _dataset_schedule(self) -> Iterator[int]:
        # Smooth weighted round robin. Equal weights reduce to ordinary round robin.
        current = [0] * len(self.weights)
        total_weight = sum(self.weights)
        for _ in range(self.schedule_length):
            for index, weight in enumerate(self.weights):
                current[index] += weight
            selected = max(range(len(current)), key=lambda index: (current[index], -index))
            current[selected] -= total_weight
            yield selected

    def __iter__(self) -> Iterator[int]:
        epoch = self.epoch
        # DataLoader constructs a new sampler iterator at each epoch. Advancing
        # here also keeps cycling correct in the existing iterator-reset path.
        self.epoch += 1

        orders = [self._shuffled_order(index, 0, epoch) for index in range(len(self.dataset_lengths))]
        positions = [0] * len(self.dataset_lengths)
        cycles = [0] * len(self.dataset_lengths)

        for dataset_index in self._dataset_schedule():
            if positions[dataset_index] >= self.dataset_lengths[dataset_index]:
                cycles[dataset_index] += 1
                orders[dataset_index] = self._shuffled_order(dataset_index, cycles[dataset_index], epoch)
                positions[dataset_index] = 0

            local_index = orders[dataset_index][positions[dataset_index]]
            positions[dataset_index] += 1
            yield self.offsets[dataset_index] + local_index
