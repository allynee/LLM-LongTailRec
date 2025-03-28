import re
from typing import List, Tuple

import pandas as pd
import wandb
import torch

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.metrics import Accuracy, NDCG_at_k, Coverage, CatalogueCoverage
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.models.rec import (
    DeepFactorizationMachine,
    ExtremeDeepFactorizationMachine,
    DeepFieldAwareFactorizationMachine,
)
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
import os

if __name__ == "__main__":
    MODEL_SAVE_PATH = "./saved_models"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    data, users, items = load_movielens100k(as_frame=True)

    # for quick inference
    subset = 1 # if we go below .3, it throws some error
    data = data.head(int(len(data) * subset))
    users = users.head(int(len(users) * subset))
    items = items.head(int(len(items) * subset))
    print(f"we have {len(data)} data points, {len(users)} users and {len(items)} items")

    list_of_genres = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # useless assertion to avoid mypy warnings
    assert (
        isinstance(items, pd.DataFrame)
        and isinstance(data, pd.DataFrame)
        and isinstance(users, pd.DataFrame)
    )
    items["genre_list"] = items[list_of_genres].apply(
        lambda x: [genre for genre in list_of_genres if x[genre] == 1], axis=1
    )

    # for each element in genre_list, all to lower case, remove non-alphanumeric
    # characters, sort and join with an underscore
    def clean_genre_list(genre_list):
        return "_".join(
            sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
        )

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, users[["user_id", "age", "gender", "occupation"]], on="user_id")
    df = pd.merge(df, items[["movie_id", "genre_list"]], on="movie_id")

    # binarize the ratings.
    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # sort by timestamp, groupby user and keep the one before the last for val
    # and the last 5 for test
    # training set got all interacts but last 6, val is the 6th, test is the last 5
    df = df.sort_values(by=["timestamp"])
    train_df = df.groupby("user_id").apply(lambda x: x.iloc[:-6]).reset_index(drop=True)
    val_df = df.groupby("user_id").apply(lambda x: x.iloc[-6]).reset_index(drop=True)
    test_df = df.groupby("user_id").apply(lambda x: x.iloc[-5:]).reset_index(drop=True)
    assert len(df) == len(train_df) + len(val_df) + len(test_df)

    cat_cols = [
        "user_id",
        "movie_id",
        "age",
        "gender",
        "occupation",
        "genre_list",
    ]

    tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, for_mf=True)
    X_tab_tr = tab_preprocessor.fit_transform(train_df)
    X_tab_val = tab_preprocessor.transform(val_df)
    X_tab_te = tab_preprocessor.transform(test_df)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
    X_wide_tr = wide_preprocessor.fit_transform(train_df)
    X_wide_val = wide_preprocessor.transform(val_df)
    X_wide_te = wide_preprocessor.transform(test_df)

    cat_embed_input: List[Tuple[str, int]] = tab_preprocessor.cat_embed_input

    models = {
        "DeepFM": DeepFactorizationMachine(
            column_idx=tab_preprocessor.column_idx, num_factors=8,
            cat_embed_input=cat_embed_input, mlp_hidden_dims=[64, 32]
        ),
        # "DeepFFM": DeepFieldAwareFactorizationMachine(
        #     column_idx=tab_preprocessor.column_idx, num_factors=8,
        #     cat_embed_input=cat_embed_input, mlp_hidden_dims=[64, 32]
        # ),
        # "XDeepFM": ExtremeDeepFactorizationMachine(
        #     column_idx=tab_preprocessor.column_idx, input_dim=16,
        #     cat_embed_input=cat_embed_input, cin_layer_dims=[32, 16],
        #     mlp_hidden_dims=[64, 32]
        # )
    }
    wide = Wide(input_dim=X_tab_tr.max(), pred_dim=1)

    # config
    BATCH_SIZE = 32
    NUM_EPOCHS = 1

    ndcg_metric = NDCG_at_k(k=5, n_cols=5)
    catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=len(list_of_genres))
    coverage_metric = Coverage(n_items_catalog=1682)  # MovieLens100k has 1682 movies

    for name, fm_model in models.items():

        wandb.init(project="idl-proj", name=f"{name}_MovieLens100k")
        wandb.config.update({
            "num_factors": 8,
            "mlp_hidden_dims": [64, 32],
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        })

        fm_model = WideDeep(wide=wide, deeptabular=fm_model)

        trainer = Trainer(
            fm_model, 
            objective="binary",
            metrics=[Accuracy()],  # Adding NDCG and Recall 
            batch_size=BATCH_SIZE, # Adding bs
        )

        X_train = {
            "X_wide": X_wide_tr,
            "X_tab": X_tab_tr,
            "target": train_df["rating"].values,
        }
        X_val = {
            "X_wide": X_wide_val,
            "X_tab": X_tab_val,
            "target": val_df["rating"].values,
        }
        X_test = {"X_wide": X_wide_te, "X_tab": X_tab_te}

        trainer.fit(X_train=X_train, X_val=X_val, n_epochs=NUM_EPOCHS)
        predictions = trainer.predict_proba(X_wide=X_wide_te, X_tab=X_tab_te)[:, 1]
        ndcg_score = ndcg_metric(torch.tensor(predictions), torch.tensor(test_df["rating"].values))


        # (1410)

        # TODO: Calculate coverage and diversity here
        # coverage needs predicted score for all users
        # input is (num_users, num_items) 
        coverage_score = coverage_metric(torch.tensor(predictions))
        print(torch.tensor(predictions))
        print(f"The torch.tensor for predictions is shape {torch.tensor(predictions).shape}")
        print(f"Inside the shape is {predictions[0]}")
        # [p(category1), p(cat2)....]
        # 1
        print(f"Coverage: {coverage_score}, NDCG@5: {ndcg_score}")
        print(f"The number of users in the test is {len(test_df['user_id'].unique())}, total number is {len(df['user_id'])}")

        # calculating coverage
        def extract_genre(input:str):
            """
            Extracts individual genres from a string of genres
            coming from the original cat data.
            Input: comedy_romance
            Output: ['comedy', 'romance']
            """
            return input.split("_")

        original_cat_data = tab_preprocessor.inverse_transform(X_tab_te)
        # convert genres from comedy_animation to [comedy, animation]
        genres = original_cat_data["genre_list"].apply(extract_genre)
        # loop through all df and then get unique genres
        unique_genres = set()
        for genre in genres:
            for g in genre:
                unique_genres.add(g)
        unique_genres = list(unique_genres)

        catalog_coverage_score = catalogue_coverage_metric(unique_genres)


        wandb.log({
            "ndcg@5": ndcg_score,
            "catalogue_coverage": catalog_coverage_score
        })

        model_path = os.path.join(MODEL_SAVE_PATH, f"{name}_model.pth")
        torch.save(fm_model.state_dict(), model_path)
        print(f"Model {name} saved at {model_path}")

        wandb.finish()


    # print(f"The cat data is \n{original_cat_data}")
    # print(f"The predictions are \n{predictions}, shape is {predictions.shape}")
