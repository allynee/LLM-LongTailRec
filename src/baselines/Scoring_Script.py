import torch

from pytorch_widedeep.metrics import Coverage_at_K_Tensor, Diversity_at_K_Tensor

NO_OF_USERS = 100
NO_OF_ITEMS = 101
POSITIVE_ITEM_IDX = 0
K = 10

tensor = torch.rand(NO_OF_USERS, NO_OF_ITEMS)
tensor[:, :50] = 0

print(tensor)

if __name__ == "__main__":
    item_coverage = Coverage_at_K_Tensor(K)
    item_coverage_score = item_coverage(tensor)

    print("Item coverage score: ", item_coverage_score)

    diversity = Diversity_at_K_Tensor(K)
    diversity_score = diversity(tensor, tensor, NO_OF_ITEMS)

    print("Diversity score: ", diversity_score)