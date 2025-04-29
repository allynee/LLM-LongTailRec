from pytorch_widedeep.metrics import Coverage_at_K_Tensor, Diversity_at_K_Tensor
from pytorch_widedeep.datasets import load_custom_data

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
    
def parse_user_and_item_ids(filename:str):
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
            if len(curr_item_ids) < 101:
                lesser_than += 1
                idxs_to_remove.append(i)
            elif len(curr_item_ids) > 101:
                greater_than += 1
            item_ids[i] = item_ids[i][-101:]

        # delete all the idxs to remove WITHOUT FUCKING UP THE ORDER
        for i in sorted(idxs_to_remove, reverse=True): # do it in reverse
            del item_ids[i]
    print(f"There ae {len(unique_user_ids)} unique users")
    return item_ids


if __name__ == "__main__":
    FILE = os.path.join(os.getcwd(), "pytorch_widedeep", "results", "101_probabilities.csv")
    K = 10

    # Test code to ensure things work
    # test_tensor = torch.rand(2, 10)
    # test_df = pd.DataFrame(test_tensor, columns=["pos","neg0", "neg1", "neg2", "neg3", "neg4", "neg5", "neg6", "neg7", "neg8", ])
    # ordered_df = get_sorted_column_names(test_df)
    # print(test_df)
    # print(ordered_df)
    # print(ordered_df.applymap(convert_column_names_to_index))
    
    df = pd.read_csv(FILE)
    print(f"OG DF shape is {df.shape}")
    ordered_df = get_sorted_column_names(df)
    print(df.head(1))

    print(ordered_df.head(1))
    ordered_df = ordered_df.applymap(convert_column_names_to_index)
    top_k_ordered_df = ordered_df.iloc[:, :K]  
    print(top_k_ordered_df.shape)
    print(top_k_ordered_df.head()) #I just want top k items

    # ============================ Calculating hit rate (Start)============================
    # Using top_k_ordered_df_head to calculate hit rate
    # it is a hit if there is a 0 for the user (0 = positive item)
    top_k_ordered_df["min"] = top_k_ordered_df.min(axis=1)
    top_k_ordered_df["hit"] = top_k_ordered_df["min"] == 0
    top_k_ordered_df["hit"] = top_k_ordered_df["hit"].astype(int)
    # Get sum of hits for all user
    sum_hr = top_k_ordered_df["hit"].sum()
    print(f"There are {sum_hr} hits and the % is {sum_hr / len(top_k_ordered_df) * 100} in the top {K} recommendations for all users.")
    # ============================ Calculating hit rate (End) ============================

    # Take bottom 101 item ids for each user
    # List of lists [ [101 item ids for user 1], [101 item ids for user 2] ... ]
    GZ_FILE = os.path.join(os.getcwd(), "pytorch_widedeep", "datasets", "custom_data_2", "amazon_movies_tv_17_Apr_dict.json.gz")
    with gzip.open(GZ_FILE, 'rb') as f:
        AMAZON_ITEM_DATA = pickle.load(f)
    TEXT_FILENAME = os.path.join(os.getcwd(), "pytorch_widedeep", "results", "final_test_data_with_negatives.txt")
    user_items_id = parse_user_and_item_ids(TEXT_FILENAME)

    # From top_k_ordered_df, for each user 

    top_k_item_indexes = []
    # def convert_relative_idx_to_item_ids(row, user_items_id):
    #     print("Helllo x2")
    #     row_id = row.name
    #     row_values = row.values
    #     print(row_id, row_values)
    #     actual_item_ids = []
    #     if row_id < len(user_items_id): # proxy fix
    #         user_item_ids = user_items_id[row_id] # index into that one
    #         for idx in row_values:
    #             actual_item_ids.append(user_item_ids[idx])
        
    def convert_relative_idx_to_item_ids(row, user_items_id):
        row_id = row.name
        # print(f"Looking at row id {row_id}")
        row_values = row.values  # make sure they're ints
        # print("Row Values:")
        # print(row_values)
        if row_id >= len(user_items_id):
            return [None] * len(row_values)  # or raise error / skip

        user_item_ids = user_items_id[row_id]
        # print(user_item_ids)
        # print(len(user_item_ids))
        return [user_item_ids[idx] for idx in row_values]
    
    print(top_k_ordered_df.head(1))
    converted_df = top_k_ordered_df.apply(lambda row: convert_relative_idx_to_item_ids(row, user_items_id), axis=1, result_type='expand')
    print(converted_df.head())

    


