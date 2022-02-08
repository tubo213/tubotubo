import multiprocessing
import os
import ssl
from copy import copy

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tubotubo.preprocessing.feature_enginnearing.base import AbstractBaseBlock


class TfidfBlock(AbstractBaseBlock):
    def __init__(self, col, dim, random_state, if_exists="pass"):
        super().__init__(if_exists)
        self.col = col
        self.dim = dim
        self.random_state = random_state
        self.pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer()),
                ("svd", TruncatedSVD(n_components=dim, random_state=random_state)),
            ]
        )

    def fit(self, input_df, y=None):
        vectorized_text = self.pipe.fit_transform(input_df[self.col].fillna("hogehoge"))

        output_df = pd.DataFrame(vectorized_text)
        output_df = output_df.add_prefix(f"Tfidf_SVD_{self.col}_")
        return output_df

    def transform(self, input_df):
        output_df = pd.DataFrame(
            self.pipe.transform(input_df[self.col].fillna("hogehoge"))
        )
        output_df = output_df.add_prefix(f"Tfidf_SVD_{self.col}_")
        return output_df


class UniversalBlock(AbstractBaseBlock):
    def __init__(self, col, dim, batch_size=256, random_state=3090, if_exists="pass"):
        super().__init__(if_exists)
        self.col = col
        self.dim = dim
        self.random_state = random_state
        self.decomp = TruncatedSVD(
            n_components=self.dim, random_state=self.random_state
        )
        self.batch_size = batch_size

        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        ssl._create_default_https_context = ssl._create_unverified_context
        url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        self.embed = hub.load(url)

    def fit(self, input_df):
        vectorized_text = self._get_vectorize_text(input_df)
        vectorized_text = self.decomp.fit_transform(vectorized_text)
        output_df = pd.DataFrame(vectorized_text).add_prefix(
            f"Universal_SVD_{self.col}"
        )
        return output_df

    def transform(self, input_df):
        vectorized_text = self._get_vectorize_text(input_df)
        vectorized_text = self.decomp.transform(vectorized_text)
        output_df = pd.DataFrame(vectorized_text).add_prefix(
            f"Universal_SVD_{self.col}"
        )
        return output_df

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

    def _get_vectorize_text(self, input_df):
        text_list = input_df[self.col].fillna("HOGEHOGE").to_list().copy()
        n_row = len(input_df)
        n_split = n_row // self.batch_size
        idx = np.linspace(0, n_row, n_split, dtype=int)

        vectorized_text = np.zeros((n_row, 512))
        for i in tqdm(range(1, n_split)):
            vectorized_text[idx[i - 1] : idx[i]] = self.embed(
                text_list[idx[i - 1] : idx[i]]
            ).numpy()
            tf.keras.backend.clear_session()
        return vectorized_text


class CustomDataset(Dataset):
    def __init__(self, df, text_col, tokenizer, max_len=128):
        self.texts = df[text_col].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer.encode(self.texts[idx])
        text_len = len(tokenized_text)

        if text_len >= self.max_len:
            inputs = tokenized_text[: self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = tokenized_text + [0] * (self.max_len - text_len)
            masks = [1] * text_len + [0] * (self.max_len - text_len)

        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        masks_tensor = torch.tensor(masks, dtype=torch.long)

        return inputs_tensor, masks_tensor


class BertBlock(AbstractBaseBlock):
    def __init__(
        self,
        dim,
        text_col,
        model,
        tokenizer,
        max_len=128,
        batch_size=16,
        random_state=3090,
        if_exists="pass",
    ):
        super().__init__(if_exists)
        self.dim = dim
        self.text_col = text_col
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.random_state = random_state
        self.decomp = TruncatedSVD(
            n_components=self.dim, random_state=self.random_state
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, input_df):
        text_emb_arr = np.zeros((len(input_df), self.dim))
        _input_df = (
            input_df.reset_index(drop=True)
            .loc[input_df[self.text_col].notnull()]
            .astype(str)
        )
        text_dataloader = self._get_dataloader(_input_df)
        _text_emb_arr = self._get_text_emb(text_dataloader)
        _text_emb_arr = self.decomp.fit_transform(_text_emb_arr)
        text_emb_arr[_input_df.index, :] = _text_emb_arr

        text_df = pd.DataFrame(text_emb_arr)
        return text_df.add_prefix("Text_")

    def transform(self, input_df):
        text_emb_arr = np.zeros((len(input_df), self.dim))
        _input_df = (
            input_df.reset_index(drop=True)
            .loc[input_df[self.text_col].notnull()]
            .astype(str)
        )
        text_dataloader = self._get_dataloader(_input_df)
        _text_emb_arr = self._get_text_emb(text_dataloader)
        _text_emb_arr = self.decomp.transform(_text_emb_arr)
        text_emb_arr[_input_df.index, :] = _text_emb_arr

        text_df = pd.DataFrame(text_emb_arr)
        return text_df.add_prefix("Text_")

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
        instance_params["model"] = instance_params["model"].__class__.__name__
        instance_params["tokenizer"] = instance_params["tokenizer"].__class__.__name__
        return instance_params

    def _get_dataloader(self, input_df):
        text_dataset = CustomDataset(
            input_df,
            text_col=self.text_col,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        text_dataloader = DataLoader(
            text_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=multiprocessing.cpu_count(),
        )
        return text_dataloader

    def _get_text_emb(self, dataloader):
        model = self.model.to(self.device).eval()
        text_emb_list = []
        for x, mask in tqdm(dataloader, desc="[Create Text Embedding Feature]"):
            x = x.to(self.device)
            mask = mask.to(self.device)
            text_emb = model(x, mask)[0]
            if text_emb.dim() == 3:
                text_emb = text_emb.mean(axis=1)
            text_emb = text_emb.to("cpu").detach().numpy().copy()
            text_emb_list.append(text_emb)

        text_emb_arr = np.concatenate(text_emb_list, axis=0)
        return text_emb_arr
