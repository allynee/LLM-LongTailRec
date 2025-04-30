import re
from typing import List, Tuple

import pandas as pd
import os


from pytorch_widedeep.metrics import Accuracy, NDCG_at_k, Coverage, CatalogueCoverage
from pytorch_widedeep.datasets import load_movielens100k, load_custom_data
import os

if __name__ == "__main__":
    FILE = os.path.join(os.getcwd(), "pytorch_widedeep", "results", "gpt2_with_metadata_scores.csv")
    K = 10
    LIST_OF_CATEGORIES = 102

    df = pd.read_csv(FILE)
    recommended_rows = (df.groupby("user_id")
                    .apply(lambda x: x.nlargest(K, "score")
                            .assign(recommended_rank=lambda df: range(1, len(df) + 1)))
                    .reset_index(drop=True))   
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):      
    #     print(recommended_rows.head())
    no_of_unique_users_test = len(df["user_id"].unique())

    # # ================= Calculating Catalogue Coverage =================
    catalogue_coverage_metric = CatalogueCoverage(n_catalogue_categories=LIST_OF_CATEGORIES)

    recommended_categories_set = set(recommended_rows["category"])
    catalogue_coverage_score = catalogue_coverage_metric(recommended_categories_set)

    # # ================= Calculating Diversity (no of categories recommended / all categories) =================

    recommended_rows_grouped_by_user_id = recommended_rows.groupby("user_id")
    user_category_sets = recommended_rows_grouped_by_user_id["category"].apply(set)
    user_unique_genres = user_category_sets.apply(len)

    user_diversity_scores = user_unique_genres / LIST_OF_CATEGORIES
    diversity = user_diversity_scores.mean()

    # # ================= Calculating Diversity (sum of rank bins ) =================
    # Rank bins : 1 (most popular), 10 (least popular)
    # [NaN, '8', '3', '9', '7', ..., '1', '2', '6', '10', '4']
    # Making NaN 10, not sure if this is intended
    recommended_rows["rank_bin_numeric"] = recommended_rows["rank_bin"].astype(str)
    recommended_rows["rank_bin_numeric"] = recommended_rows["rank_bin_numeric"].apply(
        lambda x: 20 if pd.isna(x) or x == 'nan' else float(x)
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



    print(f"Diversity will be of {K} / {LIST_OF_CATEGORIES} = {K / LIST_OF_CATEGORIES}")
    print("A lower rank_bin_avg(diversity) is indicates more popular items are recommended.")

    print(f"Coverage at {K}: {catalogue_coverage_score}")
    print(f"Diversity at {K}: {diversity}")
    print(f"Hit Rate at {K}: {percentage_users_with_recommended_in_top_k}")
    print(f"MRR at {K}: {mean_mrr}")
    print(f"Average Ranked Bin at {K}: {rank_bin_avg_avg}")
