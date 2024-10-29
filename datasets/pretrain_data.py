# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import os
import random
import warnings
from PIL import Image
import torch
import torch.utils.data
import torchvision
import kornia

from datasets.transform import resize
from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from datasets.masking_generator import TubeMaskingGenerator
from einops import rearrange


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, get_flow=False):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        if get_flow:
            assert mode == "train", "invalid: flow only for train mode"
        self.get_flow = get_flow

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        print("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(  # ./data/pretrain/train.csv
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                # print(path_label, self.cfg.DATA.PATH_LABEL_SEPARATOR, len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)))
                # exit(0)f
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(  # label:-1 path:data/pretrain/LDPolypVideo/00033.mp4
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):  # self._num_clips:1  idx:0
                    self._path_to_videos.append(
                        # _path_to_videos:list:1 ['data/pretrain/LDPolypVideo/00033.mp4'] 继续拼接 以下类推
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))  # _labels:list:1 [-1]
                    self._spatial_temporal_idx.append(idx)  # _spatial_temporal_idx:list:1 [0]
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                warnings.warn(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decode(
                container=video_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=temporal_sample_index,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                temporal_aug=self.mode == "train" and not self.cfg.DATA.NO_RGB_AUG,
                two_token=self.cfg.MODEL.TWO_TOKEN,
                rand_fr=self.cfg.DATA.RAND_FR
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            if self.mode in ["test", "val"] or self.cfg.DATA.NO_RGB_AUG:
                # Perform color normalization.
                frames = tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)

                # Perform data augmentation.
                frames = spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

                if not self.cfg.MODEL.ARCH in ['vit']:
                    frames = pack_pathway_output(self.cfg, frames)
                else:
                    # Perform temporal sampling from the fast pathway.
                    frames = torch.index_select(
                        frames,
                        1,
                        torch.linspace(
                            0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                        ).long(),
                    )

            else:
                # T H W C -> T C H W.
                frames = [rearrange(x, "t h w c -> t c h w") for x in frames]

                # Perform data augmentation.
                augmentation = VideoDataAugmentationDINO()
                frames, mask = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL,
                                      two_token=self.cfg.MODEL.TWO_TOKEN)

                # T C H W -> C T H W.
                frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

                # Perform temporal sampling from the fast pathway.
                frames = [torch.index_select(
                    x,
                    1,
                    torch.linspace(
                        0, x.shape[1] - 1, x.shape[1] if self.cfg.DATA.RAND_FR else self.cfg.DATA.NUM_FRAMES

                    ).long(),
                ) for x in frames]

            meta_data = {}

            return frames, mask,  label, index, meta_data



    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    @staticmethod
    def get_flow_from_folder(dir_path):
        flow_image_list = sorted(glob.glob(f"{dir_path}/*.jpg"))
        flow_image_list = [Image.open(im_path) for im_path in flow_image_list]
        flow_image_list = [torchvision.transforms.functional.to_tensor(im_path) for im_path in flow_image_list]
        return torch.stack(flow_image_list, dim=0)

