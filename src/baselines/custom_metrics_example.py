from pytorch_widedeep.metrics import Coverage_at_K_Tensor, Diversity_at_K_Tensor

import torch
import numpy as np

if __name__ == "__main__":
    print()
    NO_OF_ITEMS = 100
    NO_OF_USERS = 50
    NO_OF_CATEGORIES = 5

    # Testing coverage where only the top ten items are recommended
    # Answer should be 10 / NO_OF_ITEMS
    array = np.zeros((NO_OF_USERS, NO_OF_ITEMS))
    array[:, :10] = 1  # Set first 10 columns to 1
    tensor = torch.tensor(array, dtype=torch.float32)
    
    coverage = Coverage_at_K_Tensor(k=10)
    coverage_score = coverage(tensor)

    print("Coverage score for top ten items(should be 10 / NO_OF_ITEMS): ", coverage_score)
    assert coverage_score == 10 / NO_OF_ITEMS, "Coverage should be 10 / NO_OF_ITEMS"

    # Testing coverage for a random tensor
    tensor = torch.rand(NO_OF_USERS, NO_OF_ITEMS)

    coverage = Coverage_at_K_Tensor(k=10)
    coverage_score = coverage(tensor)
    print("Coverage score for random tensor(should be quite high): ", coverage_score)

    # Testing diversity for 
    # OHE categories at random for each item
    PROB_PER_CATEGORY = 0.3

    items_categories = (torch.rand(NO_OF_ITEMS, NO_OF_CATEGORIES) < PROB_PER_CATEGORY).float()
    
    # Ensure at least one category per item
    items_categories[items_categories.sum(1) == 0] = torch.tensor([1.]+[0.]*(NO_OF_CATEGORIES-1))
    print(f"Shape of items_categories should be {NO_OF_ITEMS} x {NO_OF_CATEGORIES}: {items_categories.shape}")

    diversity = Diversity_at_K_Tensor(k=10)
    diversity_score = diversity(tensor, items_categories, NO_OF_CATEGORIES)

    print("Diversity score for random tensor(should be quite high): ", diversity_score)
    print()
