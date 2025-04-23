import re
from typing import List, Tuple

import pandas as pd
import wandb
import torch

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.metrics import Accuracy, NDCG_at_k, Coverage, CatalogueCoverage
from pytorch_widedeep.datasets import load_movielens100k, load_custom_data
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
    pd.set_option('display.max_colwidth', None)

    # data, users, items = load_movielens100k(as_frame=True)
    # print("====DATA=====")
    # print(data.head())
    # print("====USERS=====")
    # print(users.head())
    # print("====ITEMS=====")
    # print(items.head())


    train_df, test_df = load_custom_data(as_frame=True, subset=0.05)
    print("\n====DATA2=====")
    print(train_df.head())
    print(train_df.columns)

    cat_cols = [
        "user_id",
        "item_id",
        "category",
        "brand",
        "rank_bin",
        "popularity_bin"
    ]

    list_of_categories = train_df["category"].unique() # genres instead of cat to match 

    tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, for_mf=True)
    X_tab_tr = tab_preprocessor.fit_transform(train_df)
    X_tab_val = tab_preprocessor.fit_transform(test_df) # duplicate
    X_tab_te = tab_preprocessor.transform(test_df)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
    X_wide_tr = wide_preprocessor.fit_transform(train_df)
    X_wide_val = wide_preprocessor.fit_transform(test_df)# duplicate
    X_wide_te = wide_preprocessor.transform(test_df)

    cat_embed_input: List[Tuple[str, int]] = tab_preprocessor.cat_embed_input

    print(f"Shape of train is {X_tab_tr.shape}")
    # Ignore for now

    # # for quick inference
    # subset = 1 # if we go below .3, it throws some error
    # data = data.head(int(len(data) * subset))
    # users = users.head(int(len(users) * subset))
    # items = items.head(int(len(items) * subset))
    # print(f"we have {len(data)} data points, {len(users)} users and {len(items)} items")

    # list_of_categories = [
    #     "unknown",
    #     "Action",
    #     "Adventure",
    #     "Animation",
    #     "Children's",
    #     "Comedy",
    #     "Crime",
    #     "Documentary",
    #     "Drama",
    #     "Fantasy",
    #     "Film-Noir",
    #     "Horror",
    #     "Musical",
    #     "Mystery",
    #     "Romance",
    #     "Sci-Fi",
    #     "Thriller",
    #     "War",
    #     "Western",
    # ]

    # # useless assertion to avoid mypy warnings
    # assert (
    #     isinstance(items, pd.DataFrame)
    #     and isinstance(data, pd.DataFrame)
    #     and isinstance(users, pd.DataFrame)
    # )
    # items["genre_list"] = items[list_of_categories].apply(
    #     lambda x: [genre for genre in list_of_categories if x[genre] == 1], axis=1
    # )

    # # for each element in genre_list, all to lower case, remove non-alphanumeric
    # # characters, sort and join with an underscore
    # def clean_genre_list(genre_list):
    #     return "_".join(
    #         sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
    #     )

    # items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    # df = pd.merge(data, users[["user_id", "age", "gender", "occupation"]], on="user_id")
    # df = pd.merge(df, items[["movie_id", "genre_list"]], on="movie_id")

    # # binarize the ratings.
    # df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    # # sort by timestamp, groupby user and keep the one before the last for val
    # # and the last 5 for test
    # # training set got all interacts but last 6, val is the 6th, test is the last 5
    # df = df.sort_values(by=["timestamp"])
    # train_df = df.groupby("user_id").apply(lambda x: x.iloc[:-6]).reset_index(drop=True)
    # val_df = df.groupby("user_id").apply(lambda x: x.iloc[-6]).reset_index(drop=True)
    # test_df = df.groupby("user_id").apply(lambda x: x.iloc[-5:]).reset_index(drop=True)
    # assert len(df) == len(train_df) + len(val_df) + len(test_df)

    # cat_cols = [
    #     "user_id",
    #     "movie_id",
    #     "age",
    #     "gender",
    #     "occupation",
    #     "genre_list",
    # ]

    # tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, for_mf=True)
    # X_tab_tr = tab_preprocessor.fit_transform(train_df)
    # X_tab_val = tab_preprocessor.transform(val_df)
    # X_tab_te = tab_preprocessor.transform(test_df)

    # wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
    # X_wide_tr = wide_preprocessor.fit_transform(train_df)
    # X_wide_val = wide_preprocessor.transform(val_df)
    # X_wide_te = wide_preprocessor.transform(test_df)

    # cat_embed_input: List[Tuple[str, int]] = tab_preprocessor.cat_embed_input

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

    # ndcg_metric = NDCG_at_k(k=5, n_cols=5) # means 5 items for all, might be wrong lol
    catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=len(list_of_categories))
    # coverage_metric = Coverage(n_items_catalog=1682)  # MovieLens100k has 1682 movies

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
            "target": train_df["label"].values,
        }

        X_val = {
            "X_wide": X_wide_val,
            "X_tab": X_tab_val,
            "target": test_df["label"].values,
        }

        X_test = {"X_wide": X_wide_te, "X_tab": X_tab_te}

        trainer.fit(X_train=X_train, X_val=X_val, n_epochs=NUM_EPOCHS)
        predictions = trainer.predict_proba(X_wide=X_wide_te, X_tab=X_tab_te)[:, 1]

        # ndcg_score = ndcg_metric(torch.tensor(predictions), torch.tensor(test_df["label"].values))
        
        # ================= Calculating Coverage =================
        threshold = 0.5

        positive_idx = predictions >= threshold 

        original_cat_data = tab_preprocessor.inverse_transform(X_tab_te)
        recommended_rows = original_cat_data[positive_idx]

        catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=len(list_of_categories))
        recommended_categories_set = set(recommended_rows["category"])
        catalogue_coverage_score = catalogue_coverage_metric(recommended_categories_set)

        # ================= Calculating Diversity =================

        recommended_rows_grouped_by_user_id = recommended_rows.groupby("user_id")
        print(recommended_rows_grouped_by_user_id.head())

        # create set of categories for each user
        user_category_sets = recommended_rows_grouped_by_user_id["category"].apply(set)

        # calculate the number of unique genres for each user
        user_unique_genres = user_category_sets.apply(len)

        # calculate the diversity score for each user
        user_diversity_scores = user_unique_genres / len(list_of_categories)

        # calculate the average diversity score across all users
        diversity = user_diversity_scores.mean()

        wandb.log({
            # "ndcg@5": ndcg_score,
            "catalogue_coverage": catalogue_coverage_score,
            "diversity": diversity
        })

        model_path = os.path.join(MODEL_SAVE_PATH, f"{name}_model.pth")
        torch.save(fm_model.state_dict(), model_path)
        print(f"Model {name} saved at {model_path}")

        wandb.finish()
