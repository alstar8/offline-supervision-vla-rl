import os
from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import concurrent.futures
import tensorflow_datasets as tfds

import numpy as np


def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]

        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        # Preserve gripper toggle events (e.g., from -1 to 1 or vice versa)
        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid

    return valid_mask

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    DEFAULT_TOTAL_DEMOS = 8200
    DEFAULT_VAL_DEMOS = 16
    DEFAULT_SOURCE_DATA_DIR = "../../../ManiSkill/mp_collect/PutOnPlateInScene25Main-v3/8200/data"
    DEFAULT_BUFFER_SIZE = 50
    DEFAULT_MAX_WORKERS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_demos = int(os.environ.get("RLVLA_SFT_TOTAL_DEMOS", self.DEFAULT_TOTAL_DEMOS))
        self.val_demos = int(os.environ.get("RLVLA_SFT_VAL_DEMOS", self.DEFAULT_VAL_DEMOS))
        self.source_data_dir = os.environ.get("RLVLA_SFT_SOURCE_DATA_DIR", self.DEFAULT_SOURCE_DATA_DIR)
        self.buffer_size = int(os.environ.get("RLVLA_SFT_BUFFER_SIZE", self.DEFAULT_BUFFER_SIZE))
        self.max_workers = int(os.environ.get("RLVLA_SFT_MAX_WORKERS", self.DEFAULT_MAX_WORKERS))
        self.tasks = [
            {"name": self.source_data_dir},
        ]
        if self.total_demos <= 0:
            raise ValueError(f"RLVLA_SFT_TOTAL_DEMOS must be positive, got {self.total_demos}.")
        if self.val_demos < 0:
            raise ValueError(f"RLVLA_SFT_VAL_DEMOS must be non-negative, got {self.val_demos}.")
        if self.buffer_size <= 0:
            raise ValueError(f"RLVLA_SFT_BUFFER_SIZE must be positive, got {self.buffer_size}.")
        if self.max_workers <= 0:
            raise ValueError(f"RLVLA_SFT_MAX_WORKERS must be positive, got {self.max_workers}.")
        print(
            "SFT builder config | "
            f"source_data_dir={self.source_data_dir} | "
            f"total_demos={self.total_demos} | "
            f"val_demos={self.val_demos} | "
            f"buffer_size={self.buffer_size} | "
            f"max_workers={self.max_workers}"
        )

    def _discover_episode_files(self):
        all_files = []
        for task in self.tasks:
            path = Path(task["name"])
            files = sorted(glob.glob(str(path / "*.npz")))
            all_files.extend(files)
            print(f"Found {len(files)} files in {path}")

        if len(all_files) < self.total_demos:
            print(
                f"Warning: expected {self.total_demos} demos but found {len(all_files)}; "
                "building dataset from available files."
            )
        return all_files[:self.total_demos]

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Observation image.'
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32, ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        all_files = self._discover_episode_files()
        if len(all_files) <= self.val_demos:
            raise ValueError(
                f"Not enough episodes ({len(all_files)}) for val split size {self.val_demos}."
            )
        train_files = all_files[:-self.val_demos]
        val_files = all_files[-self.val_demos:]
        print(f"Split sizes -> train: {len(train_files)}, val: {len(val_files)}")
        return {
            'train': self._generate_examples(train_files),
            'val': self._generate_examples(val_files),
        }

    def _generate_examples(self, all_files) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            data = np.load(episode_path, allow_pickle=True)["arr_0"].tolist()

            # prepare data
            ins = data['instruction']
            ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
            actions = data["action"]
            images = np.asarray([np.asarray(img) for img in data["image"]])

            mask = filter_small_actions(data["action"])
            actions = actions[mask]
            images = images[mask]
            num_filtered = mask.shape[0] - mask.sum()
            print(f"Filtered {num_filtered}/{mask.shape[0]} actions")

            episode = []
            success_count = 0
            for i in range(len(actions)):
                episode.append({
                    'observation': {
                        'image': images[i],
                    },
                    'action': actions[i],
                    'language_instruction': ins,
                })

                if data["info"][i]["success"]:
                    success_count += 1
                else:
                    success_count = 0

                if success_count >= 6:
                    break

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
            del data

            return sample, num_filtered

        print(f"{len(all_files)}")



        buffer_size = self.buffer_size
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        futures = {}
        it = iter(all_files)
        for _ in range(buffer_size):
            try:
                path = next(it)
                futures[executor.submit(_parse_example, path)] = path
            except StopIteration:
                break

        while futures:
            done, _ = concurrent.futures.wait(
                list(futures.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                ep_path = futures.pop(future)
                sample, num_filtered = future.result()
                yield ep_path, sample

                try:
                    next_path = next(it)
                    futures[executor.submit(_parse_example, next_path)] = next_path
                except StopIteration:
                    pass

        executor.shutdown()
