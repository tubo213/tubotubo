import os
import ssl
from copy import copy

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
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
