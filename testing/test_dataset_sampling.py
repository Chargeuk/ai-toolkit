import unittest
from unittest.mock import patch

from torch.utils.data import ConcatDataset, Dataset, RandomSampler

from toolkit.config_modules import DatasetConfig, TrainConfig
from toolkit.data_loader import get_dataloader_from_datasets, get_dataloader_datasets
from toolkit.dataset_sampling import DatasetRoundRobinSampler, normalize_dataset_sampling_strategy


class TinyDataset(Dataset):
    def __init__(self, config, batch_size=1, sd=None):
        self.dataset_config = config
        self.length = 3 if config.folder_path.endswith('a') else 2

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class DatasetRoundRobinSamplerTests(unittest.TestCase):
    def selected_datasets(self, sampler):
        offsets = sampler.offsets + [sum(sampler.dataset_lengths)]
        return [next(i for i in range(len(sampler.dataset_lengths)) if offsets[i] <= value < offsets[i + 1])
                for value in sampler]

    def test_equal_round_robin_order(self):
        sampler = DatasetRoundRobinSampler([5, 5, 5])
        self.assertEqual(self.selected_datasets(sampler), [0, 1, 2] * 5)

    def test_weighted_three_to_one_is_smooth_and_exact(self):
        sampler = DatasetRoundRobinSampler([8, 8], weights=[3, 1])
        schedule = self.selected_datasets(sampler)
        self.assertEqual(schedule, [0, 0, 1, 0] * 4)
        self.assertEqual(schedule.count(0), 12)
        self.assertEqual(schedule.count(1), 4)

    def test_child_dataset_reshuffles_and_cycles(self):
        sampler = DatasetRoundRobinSampler([5, 1], weights=[10, 1])
        sampled = list(sampler)
        dataset_zero = [value for value in sampled if value < 5]
        self.assertEqual(set(dataset_zero[:5]), set(range(5)))
        self.assertEqual(set(dataset_zero[5:10]), set(range(5)))

    def test_validation(self):
        for invalid in ([0, 1], [-1, 1], [1.5, 1], [True, 1]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    DatasetRoundRobinSampler([2, 2], weights=invalid)
        with self.assertRaises(ValueError):
            normalize_dataset_sampling_strategy('random_magic')
        with self.assertRaises(ValueError):
            DatasetConfig(folder_path='a', sampling_weight=0)
        with self.assertRaises(ValueError):
            TrainConfig(dataset_sampling_strategy='random_magic')

    @patch('toolkit.data_loader.AiToolkitDataset', TinyDataset)
    def test_default_loader_keeps_concat_shuffle_behavior(self):
        configs = [
            DatasetConfig(folder_path='a', buckets=False, num_workers=0),
            DatasetConfig(folder_path='b', buckets=False, num_workers=0),
        ]
        loader = get_dataloader_from_datasets(configs, batch_size=1)
        self.assertIsInstance(loader.dataset, ConcatDataset)
        self.assertIsInstance(loader.sampler, RandomSampler)

    @patch('toolkit.data_loader.AiToolkitDataset', TinyDataset)
    def test_regularization_groups_remain_separate(self):
        normal = [DatasetConfig(folder_path='a', buckets=False, num_workers=0, is_reg=False)]
        regularization = [DatasetConfig(folder_path='b', buckets=False, num_workers=0, is_reg=True)]
        normal_loader = get_dataloader_from_datasets(normal, dataset_sampling_strategy='round_robin')
        reg_loader = get_dataloader_from_datasets(regularization, dataset_sampling_strategy='round_robin')
        self.assertTrue(all(not dataset.dataset_config.is_reg for dataset in get_dataloader_datasets(normal_loader)))
        self.assertTrue(all(dataset.dataset_config.is_reg for dataset in get_dataloader_datasets(reg_loader)))


if __name__ == '__main__':
    unittest.main()
