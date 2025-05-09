from pytorch_widedeep.metrics import Coverage_at_K_Tensor, Diversity_at_K_Tensor
from pytorch_widedeep.datasets import load_custom_data, load_item_data

import torch
import gzip
import json
import numpy as np
import pandas as pd
import os
import pickle

def get_sorted_column_names(df):
    """
    Converts from 
                pos      neg0  neg1  neg2  neg3  ...  neg95  neg96  neg97  neg98  neg99
    user id 

    to 
                0, 1, 2, 3, ... n
                neg5, pos, neg3....
    user id
    """
    sorted_column_names = []
    for _, row in df.iterrows():
        sorted_cols = row.sort_values(ascending=False).index.tolist()
        sorted_column_names.append(sorted_cols)
    return pd.DataFrame(sorted_column_names)

def convert_column_names_to_index(input:str):
    """
    pos = 0
    neg0 = 1
    neg1 = 2
    """
    if input == "pos":
        return 0
    else:
        return int(input[3:]) + 1
    
def parse_user_and_item_ids(filename:str, SELECTED_CANDIDATES:int):
    """
    This parses and extracts only last 101 ids for each user
    """
    
    #extract the last 101 ids for each user
    # open text file
    item_ids = []
    c_user_id = -1
    idx = -1
    unique_user_ids = set()

    greater_than = 0
    lesser_than = 0

    with open(filename) as f:
        for line in f:
            # Print each line
            user_id, item_id = line.strip().split(" ")
            item_id = int(item_id)
            if not (user_id == c_user_id):
                idx += 1
                c_user_id = user_id
                item_ids.append([])

            item_ids[idx].append(item_id)
            unique_user_ids.add(user_id)
        
        idxs_to_remove  = []

        for i in range(len(item_ids)):
            curr_item_ids = item_ids[i]
            if len(curr_item_ids) < TOTAL_CANDIDATES:
                lesser_than += 1
                idxs_to_remove.append(i)
            elif len(curr_item_ids) > TOTAL_CANDIDATES:
                greater_than += 1
            item_ids[i] = item_ids[i][-TOTAL_CANDIDATES:]
            item_ids[i] = item_ids[i][:SELECTED_CANDIDATES]

        # delete all the idxs to remove WITHOUT FUCKING UP THE ORDER
        for i in sorted(idxs_to_remove, reverse=True): # do it in reverse
            del item_ids[i]
    return item_ids


if __name__ == "__main__":
    FILE = os.path.join(os.getcwd(), "pytorch_widedeep", "results", "71_probabilities.csv")
    K = 10
    TOTAL_CANDIDATES = 101
    SELECTED_CANDIDATES = 71

    # Test code to ensure things work
    # test_tensor = torch.rand(2, 10)
    # test_df = pd.DataFrame(test_tensor, columns=["pos","neg0", "neg1", "neg2", "neg3", "neg4", "neg5", "neg6", "neg7", "neg8", ])
    # ordered_df = get_sorted_column_names(test_df)
    # print(test_df)
    # print(ordered_df)
    # print(ordered_df.applymap(convert_column_names_to_index))
    
    df = pd.read_csv(FILE)
    ordered_df = get_sorted_column_names(df)

    ordered_df = ordered_df.applymap(convert_column_names_to_index)
    top_k_ordered_df = ordered_df.iloc[:, :K]  

    # ============================ Calculating hit rate (Start)============================
    # Calculate mean recriprocal rank
    # Using top_k_ordered_df_head to calculate hit rate
    # it is a hit if there is a 0 for the user (0 = positive item)
    mask = top_k_ordered_df == 0
    top_k_ordered_df["first_hit_position"] = mask.values.argmax(axis=1)
    top_k_ordered_df.loc[~mask.any(axis=1), "first_hit_position"] = -1
    def get_mrr(row):
        if row["first_hit_position"] == -1:
            return 0
        else:
            return 1 / (row["first_hit_position"] + 1)
        
    top_k_ordered_df["mrr"] = top_k_ordered_df.apply(get_mrr, axis=1)

    AVG_MRR = top_k_ordered_df["mrr"].mean()

    # Hit rate
    top_k_ordered_df["min"] = top_k_ordered_df.min(axis=1)
    top_k_ordered_df["hit"] = top_k_ordered_df["min"] == 0
    top_k_ordered_df["hit"] = top_k_ordered_df["hit"].astype(int)
    # Get sum of hits for all user
    sum_hr = top_k_ordered_df["hit"].sum()
    HIT_RATE = sum_hr / len(top_k_ordered_df)
    # remove the hit and min to avoid it fking up later
    # ============================ Calculating MRR rate (End) ============================


    top_k_ordered_df = top_k_ordered_df.drop(["hit", "min", "first_hit_position", "mrr"], axis=1)
    # ============================ Calculating hit rate (End) ============================

    # Take bottom 101 item ids for each user
    # List of lists [ [101 item ids for user 1], [101 item ids for user 2] ... ]

    TEXT_FILENAME = os.path.join(os.getcwd(), "pytorch_widedeep", "results", "final_test_data_with_negatives.txt")
    user_items_id = parse_user_and_item_ids(TEXT_FILENAME, SELECTED_CANDIDATES)

    # From top_k_ordered_df, for each user 
    top_k_item_indexes = []

    def convert_relative_idx_to_item_ids(row, user_items_id):
        row_id = row.name
        row_values = row.values  # make sure they're ints
        if row_id >= len(user_items_id):
            return [None] * len(row_values)  # or raise error / skip

        user_item_ids = user_items_id[row_id]
        return [user_item_ids[idx] for idx in row_values]
    
    converted_df = top_k_ordered_df.apply(lambda row: convert_relative_idx_to_item_ids(row, user_items_id), axis=1, result_type='expand')

    # ============================ Calculating category coverage and diversity (Start) ============================
    item_data = load_item_data(as_frame=True)

    # create dict mapping of item id to category
    item_to_category = {}
    item_to_ranked_bin = {}
    for index, row in item_data.iterrows():
        item_to_category[row["item_id"]] = row["category"]
        item_to_ranked_bin[row["item_id"]] = row["rank_bin"]

    def item_id_to_category(item_id):
        if item_id in item_to_category:
            return item_to_category[item_id]
        else:
            return "others"
    def get_avg_user_ranked_bin(row):
        return sum(
            10 if pd.isna(item_to_ranked_bin.get(item_id, float('nan'))) 
            else int(item_to_ranked_bin[item_id]) 
            for item_id in row
        ) / len(row)

    def no_of_unique_categories_per_row(row):
        return len(set(row))
    
    converted_df["avg_ranked_bin"] = converted_df.apply(get_avg_user_ranked_bin, axis=1)
    AVG_RANKED_BIN = converted_df["avg_ranked_bin"].mean()
    converted_df.drop(["avg_ranked_bin"], axis=1, inplace=True)


    converted_df_to_category = converted_df.applymap(item_id_to_category)
    converted_df_to_category["no_of_unique_categories"] = converted_df_to_category.apply(no_of_unique_categories_per_row, axis=1)
    converted_df_to_category["diversity"] = converted_df_to_category["no_of_unique_categories"] / 102

    DIVERSITY = converted_df_to_category["diversity"].mean()
    # Get all unique categories
    unique_categories = set()
    for index, row in item_data.iterrows():
        unique_categories.add(row["category"])
    COVERAGE = len(unique_categories) / 102
    # ============================ Calculating category coverage and diversity (End) ============================

    # Calculating ranked bin diversity
    # ==== PRINT METRICS ====
    print(f"Coverage at {K}: {COVERAGE}")
    print(f"Diversity at {K}: {DIVERSITY}")
    print(f"Hit Rate at {K}: {HIT_RATE}")
    print(f"MRR at {K}: {AVG_MRR}")
    print(f"Average Ranked Bin at {K}: {AVG_RANKED_BIN}")
