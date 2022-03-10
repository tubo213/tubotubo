import multiprocessing
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm.notebook import tqdm
from tubotubo.preprocessing.feature_enginnearing.base import AbstractBaseBlock
from typing_extensions import Literal


class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, path_col="img_path", target_col="label", image_size=224
    ):
        self._X = df[path_col].values
        self._y = None
        if target_col in df.keys():
            self._y = df[target_col].values
        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image


class IMGBlock(AbstractBaseBlock):
    def __init__(
        self,
        dim: int,
        path_col: str,
        batch_size: int = 64,
        backbone: str = "resnet34",
        image_size: int = 224,
        in_chans: Literal[1, 2, 3] = 3,
        random_state: int = 3090,
        if_exists: Literal["pass", "repalce"] = "pass",
    ):
        super().__init__(if_exists)
        self.dim = dim
        self.path_col = path_col
        self.random_state = random_state
        self.batch_size = batch_size
        self.backbone = backbone
        self.image_size = image_size
        self.decomp = TruncatedSVD(
            n_components=self.dim, random_state=self.random_state
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_chans = in_chans
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transformer = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean[: self.in_chans], std=std[: self.in_chans]),
            ]
        )

    def fit(self, input_df: pd.DataFrame):
        _input_df = input_df.loc[
            input_df[self.path_col].apply(lambda x: Path(x).exists())
        ].copy()
        img_dataloader = self._get_dataloader(_input_df)
        img_emb_arr = self._get_image_emb(img_dataloader)
        img_emb_arr = self.decomp.fit_transform(img_emb_arr)

        _img_df = pd.DataFrame(img_emb_arr, index=_input_df[self.path_col]).add_prefix(
            f"IMG_{self.backbone}_"
        )
        img_df = (
            input_df[[self.path_col]]
            .merge(_img_df, how="left", on=self.path_col)
            .drop(self.path_col, axis=1)
        )
        return img_df

    def transform(self, input_df):
        _input_df = input_df.loc[
            input_df[self.path_col].apply(lambda x: Path(x).exists())
        ].copy()
        img_dataloader = self._get_dataloader(_input_df)
        img_emb_arr = self._get_image_emb(img_dataloader)
        img_emb_arr = self.decomp.transform(img_emb_arr)

        _img_df = pd.DataFrame(img_emb_arr, index=_input_df[self.path_col]).add_prefix(
            f"IMG_{self.backbone}_"
        )
        img_df = (
            input_df[[self.path_col]]
            .merge(_img_df, how="left", on=self.path_col)
            .drop(self.path_col, axis=1)
        )
        return img_df

    def get_init_params(self) -> dict:
        init_param_names = self.__init__.__code__.co_varnames[
            1 : self.__init__.__code__.co_argcount
        ]
        instance_params = copy(self.__dict__)
        instance_param_names = list(instance_params.keys())
        for key in instance_param_names:
            if key not in init_param_names:
                del instance_params[key]

        del instance_params["batch_size"]
        return instance_params

    def _get_dataloader(self, input_df):
        img_dataset = CustomDataset(
            input_df, path_col=self.path_col, image_size=self.image_size
        )
        img_dataloader = DataLoader(
            img_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=multiprocessing.cpu_count(),
        )
        return img_dataloader

    def _get_image_emb(self, dataloader):
        model = (
            timm.create_model(
                self.backbone, num_classes=0, in_chans=self.in_chans, pretrained=True
            )
            .to(self.device)
            .eval()
        )

        image_emb_list = []
        for image in tqdm(dataloader, desc="[Create Image Embedding Feature]"):
            image = self.transformer(image).to(self.device)
            image_emb = model(image)
            image_emb = image_emb.to("cpu").detach().numpy().copy()
            image_emb_list.append(image_emb)
        image_emb_arr = np.concatenate(image_emb_list, axis=0)
        return image_emb_arr
