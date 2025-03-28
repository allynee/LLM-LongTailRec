def example_ndcg():
    ndcg = NDCG_at_k(k=10)
    y_pred = torch.rand(100, 5)
    y_true = torch.randint(2, (100,))
    score = ndcg(y_pred, y_true)
    print(score)

def example_coverage():
    coverage = Coverage(k=10, n_items_catalog=100)
    y_pred = torch.rand(100, 5) 
    y_true = torch.randint(2, (100,))
    score = coverage(y_pred, y_true)
    print(score)

def example_diversity():
    diversity = Diversity(k=10, item_features=torch.rand(100, 20))
    y_pred = torch.rand(100, 5)
    y_true = torch.randint(2, (100,))
    score = diversity(y_pred, y_true)
    print(score)
def main():
    pass

if __name__ == '__main__':
    import torch
    from metrics import NDCG_at_k, Coverage, Diversity

    example_ndcg()
    example_coverage()
    example_diversity()