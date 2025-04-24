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

  

    train_df, val_df, test_df = load_custom_data(as_frame=True, subset=0.01)

    cat_cols = [
        "user_id",
        "item_id",
        "category",
        "brand",
        "rank_bin",
        "popularity_bin"
    ]

    no_of_unique_users_test = len(test_df["user_id"].unique())
    list_of_categories = test_df["category"].unique() # genres instead of cat to match 

    tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_cols, for_mf=True)
    X_tab_tr = tab_preprocessor.fit_transform(train_df)
    X_tab_val = tab_preprocessor.fit_transform(val_df) # duplicate
    X_tab_te = tab_preprocessor.transform(test_df)

    wide_preprocessor = WidePreprocessor(wide_cols=cat_cols)
    X_wide_tr = wide_preprocessor.fit_transform(train_df)
    X_wide_val = wide_preprocessor.fit_transform(val_df)# duplicate
    X_wide_te = wide_preprocessor.transform(test_df)

    cat_embed_input: List[Tuple[str, int]] = tab_preprocessor.cat_embed_input

    models = {
        "DeepFM": DeepFactorizationMachine(
            column_idx=tab_preprocessor.column_idx, num_factors=8,
            cat_embed_input=cat_embed_input, mlp_hidden_dims=[64, 32]
        ),
        "DeepFFM": DeepFieldAwareFactorizationMachine(
            column_idx=tab_preprocessor.column_idx, num_factors=8,
            cat_embed_input=cat_embed_input, mlp_hidden_dims=[64, 32]
        ),
        "XDeepFM": ExtremeDeepFactorizationMachine(
            column_idx=tab_preprocessor.column_idx, input_dim=16,
            cat_embed_input=cat_embed_input, cin_layer_dims=[32, 16],
            mlp_hidden_dims=[64, 32]
        )
    }
    wide = Wide(input_dim=X_tab_tr.max(), pred_dim=1)

    # config
    BATCH_SIZE = 32
    NUM_EPOCHS = 1

    # ndcg_metric = NDCG_at_k(k=5, n_cols=5) # means 5 items for all, might be wrong lol
    catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=len(list_of_categories))
    # coverage_metric = Coverage(n_items_catalog=1682)  # MovieLens100k has 1682 movies

    for name, fm_model in models.items():

        wandb.init(project="idl-proj-actual-training", name=f"{name}_MovieLens100k")
        # wandb.init(project="idl-proj", name=f"{name}_MovieLens100k")
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

        # here, i have individual user ids : item id and probability
        # i join the probability to the original one
        # I get user id: item id : probaiblity
        # ndcg_score = ndcg_metric(torch.tensor(predictions), torch.tensor(test_df["label"].values))
        
        # ================= Calculating Coverage =================

        assert(len(predictions) == len(test_df))
        joined_df = test_df
        # Concate them together
        joined_df["probability"] = predictions

        # Coverage at K, NDCG at K, Diversity at K
        K = 10

        # Now, we group by user_id and get top k highest probability
        # in the event of ties, they will still rank differently
        recommended_rows = (joined_df.groupby("user_id")
                        .apply(lambda x: x.nlargest(K, "probability")
                                .assign(recommended_rank=lambda df: range(1, len(df) + 1)))
                        .reset_index(drop=True))     
        # # ================= Calculating Catalogue Coverage =================

        catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=len(list_of_categories))
        recommended_categories_set = set(recommended_rows["category"])
        catalogue_coverage_score = catalogue_coverage_metric(recommended_categories_set)

        # # ================= Calculating Diversity (no of categories recommended / all categories) =================

        recommended_rows_grouped_by_user_id = recommended_rows.groupby("user_id")
        user_category_sets = recommended_rows_grouped_by_user_id["category"].apply(set)
        user_unique_genres = user_category_sets.apply(len)

        user_diversity_scores = user_unique_genres / len(list_of_categories)
        diversity = user_diversity_scores.mean()

        # # ================= Calculating Diversity (sum of rank bins ) =================
        # Rank bins : 1 (most popular), 10 (least popular)
        # [NaN, '8', '3', '9', '7', ..., '1', '2', '6', '10', '4']
        # Making NaN 10, not sure if this is intended
        recommended_rows["rank_bin_numeric"] = recommended_rows["rank_bin"].astype(str)
        recommended_rows["rank_bin_numeric"] = recommended_rows["rank_bin_numeric"].apply(
            lambda x: 20 if pd.isna(x) or x == 'nan' else int(x)
        )
        recommended_rows_grouped_by_user_id = recommended_rows.groupby("user_id")
        user_ranked_bin_avg = recommended_rows_grouped_by_user_id["rank_bin_numeric"].mean()

        rank_bin_avg_avg = user_ranked_bin_avg.mean()

        # ================= Calculating Mean Reciprocal Rank =================  
        # This is supposed to take a look at the top k items recommended to each user
        # It identifies if there is an item in there with the label == 1 (positive item)
        # If there is, it takes the ranking of the probability of this item 
        # relative to all items recommended. Eg the positive item has the 2nd highest probability
        # It calculates MRR by doing 1/rank, in this case 2
        # If there is no item with label == 1, it returns 0

        # Find the columns with 1 in the label
        # Divide 1 / recommended rank
        def calculate_mrr(user_group):
            # 1 / recommended rank, else 0
            positive_items = user_group[user_group['label'] == 1]
            
            if len(positive_items) == 0:
                return 0.0
            
            best_rank = positive_items['recommended_rank'].min()
            
            return 1.0 / best_rank if best_rank > 0 else 0.0
        
        user_mrr_scores = recommended_rows.groupby('user_id').apply(calculate_mrr)
        mean_mrr = user_mrr_scores.mean()

        # number of not zeros
        users_with_recommended_in_top_k = user_mrr_scores[user_mrr_scores != 0].count()
        percentage_users_with_recommended_in_top_k = users_with_recommended_in_top_k / no_of_unique_users_test


        wandb.log({
            # "ndcg@5": ndcg_score,
            "mrr": mean_mrr,
            "catalogue_coverage": catalogue_coverage_score,
            "diversity": diversity,
            "rank_bin_avg(diversity)": rank_bin_avg_avg, # Lower number is better
            "percentage_users_with_recommended_in_top_k": percentage_users_with_recommended_in_top_k
        })

        print(f"Diversity will be of {K} / {len(list_of_categories)} = {K / len(list_of_categories)}")
        print("A lower rank_bin_avg(diversity) is indicates more popular items are recommended.")

        model_path = os.path.join(MODEL_SAVE_PATH, f"{name}_model.pth")
        torch.save(fm_model.state_dict(), model_path)
        print(f"Model {name} saved at {model_path}")

        wandb.finish()
