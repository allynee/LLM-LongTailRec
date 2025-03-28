import numpy as np
import torch
from typing import List, Dict, Set, Union, Optional
from torch import Tensor

def reshape_to_2d(tensor: Tensor, n_columns: int) -> Tensor:
    if tensor.dim() == 1:
        if tensor.size(0) % n_columns != 0:
            raise ValueError(
                f"Tensor length ({tensor.size(0)}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = tensor.size(0) // n_columns
        return tensor.reshape(n_rows, n_columns)
    elif tensor.dim() == 2 and tensor.size(1) == 1:
        if tensor.size(0) % n_columns != 0:
            raise ValueError(
                f"Tensor length ({tensor.size(0)}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = tensor.size(0) // n_columns
        return tensor.reshape(n_rows, n_columns)
    else:
        raise ValueError(
            "Input tensor must be 1-dimensional or 2-dimensional with one column"
        )

class Metric(object):
    def __init__(self):
        self._name = ""

    def reset(self):
        raise NotImplementedError("Custom Metrics must implement this function")

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        raise NotImplementedError("Custom Metrics must implement this function")


class MultipleMetrics(object):
    def __init__(self, metrics: List[Union[Metric, object]], prefix: str = ""):
        instantiated_metrics = []
        for metric in metrics:
            if isinstance(metric, type):
                instantiated_metrics.append(metric())
            else:
                instantiated_metrics.append(metric)
        self._metrics = instantiated_metrics
        self.prefix = prefix

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Dict:
        logs = {}
        for metric in self._metrics:
            if isinstance(metric, Metric):
                logs[self.prefix + metric._name] = metric(y_pred, y_true)
            elif isinstance(metric):
                metric.update(y_pred, y_true.int())  # type: ignore[attr-defined]
                logs[self.prefix + type(metric).__name__] = (
                    metric.compute().detach().cpu().numpy()
                )
        return logs

class CatalogueCoverage(Metric):
    def __init__(self, n_catalogue_categories: int):
        super(CatalogueCoverage, self).__init__()
        self.n_catalogue_categories = n_catalogue_categories

    def __call__(self, categories:Set):
        return len(categories) / self.n_catalogue_categories

class Coverage(Metric):
    r"""
    Coverage metric measures the percentage of items that are recommended at least once.
    
    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. 
    k: int, Optional, default = None

    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import Coverage
    >>>
    >>> coverage = Coverage(k=10, n_items_catalog=100)
    >>> y_pred = torch.rand(100, 5)
    >>> y_true = torch.randint(2, (100,))
    >>> score = coverage(y_pred, y_true)
    """
    def __init__(
        self, 
        n_cols: int = 10, 
        k: Optional[int] = None, 
        n_items_catalog: Optional[int] = None
    ):
        super(Coverage, self).__init__()
        
        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )
        
        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self.n_items_catalog = n_items_catalog
        self._name = f"coverage@{k}"
        self.reset()
    
    def reset(self):
        self.recommended_items = None
        self.count = 0
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        device = y_pred.device
        
        if y_pred.ndim > 1 and y_pred.size(1) > 1:
            # multiclass
            y_pred = y_pred.topk(1, 1)[1]
        
        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        
        batch_size = y_pred_2d.shape[0]
        
        # Get the top-k items for each user/query
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        
        # If n_items_catalog is not provided, infer it from the data
        if self.n_items_catalog is None:
            self.n_items_catalog = int(y_pred_2d.max().item()) + 1
        
        # Update the set of recommended items
        if self.recommended_items is None:
            self.recommended_items = set(top_k_indices.cpu().numpy().flatten())
        else:
            self.recommended_items.update(top_k_indices.cpu().numpy().flatten())
        
        # Calculate coverage
        coverage = len(self.recommended_items) / self.n_items_catalog
        
        self.count += batch_size
        
        return np.array(coverage)


class Diversity(Metric):
    r"""
    Diversity metric measures the average pairwise distance between recommended items.
    
    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor.
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.
    item_features: Optional[Tensor], default = None
        Features for each item in the catalog. If provided, the diversity will be
        calculated based on the cosine distance between item features.
        Shape: (n_items_catalog, n_features)
    distance_metric: str, default = 'cosine'
        Distance metric to use. Can be 'cosine', 'euclidean', or 'hamming'.
    
    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import Diversity
    >>>
    >>> # Create item features matrix (100 items with 20 features each)
    >>> item_features = torch.rand(100, 20)
    >>> diversity = Diversity(k=10, item_features=item_features)
    >>> y_pred = torch.rand(100, 5)
    >>> y_true = torch.randint(2, (100,))
    >>> score = diversity(y_pred, y_true)
    """
    def __init__(
        self, 
        n_cols: int = 10, 
        k: Optional[int] = None, 
        item_features: Optional[Tensor] = None,
        distance_metric: str = 'cosine'
    ):
        super(Diversity, self).__init__()
        
        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )
        
        if distance_metric not in ['cosine', 'euclidean', 'hamming']:
            raise ValueError(
                f"distance_metric must be one of ['cosine', 'euclidean', 'hamming']. Got {distance_metric}"
            )
        
        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self.item_features = item_features
        self.distance_metric = distance_metric
        self._name = f"diversity@{k}"
        self.reset()
    
    def reset(self):
        self.sum_diversity = 0.0
        self.count = 0
    
    def _compute_pairwise_distance(self, indices: Tensor) -> Tensor:
        """
        Compute the pairwise distance between items based on their features.
        
        Parameters
        ----------
        indices: Tensor
            Indices of the top-k items. Shape: (batch_size, k)
            
        Returns
        -------
        Tensor
            Pairwise distances between items. Shape: (batch_size,)
        """
        device = indices.device
        batch_size, k = indices.shape
        
        if self.item_features is None:
            # If no item features are provided, return a constant diversity
            return torch.ones(batch_size, device=device)
        
        # Get features for the selected items
        features = self.item_features.to(device)[indices]  # Shape: (batch_size, k, n_features)
        
        total_distances = torch.zeros(batch_size, device=device)
        
        for i in range(k):
            for j in range(i + 1, k):
                if self.distance_metric == 'cosine':
                    # Compute cosine similarity and convert to distance
                    similarity = torch.nn.functional.cosine_similarity(
                        features[:, i], features[:, j], dim=1
                    )
                    distance = 1 - similarity
                elif self.distance_metric == 'euclidean':
                    # Compute Euclidean distance
                    distance = torch.norm(features[:, i] - features[:, j], dim=1)
                else:  # hamming
                    # Compute Hamming distance (assuming binary features or converting to binary)
                    binary_i = (features[:, i] > 0.5).float()
                    binary_j = (features[:, j] > 0.5).float()
                    distance = torch.sum(binary_i != binary_j, dim=1) / features.size(2)
                
                total_distances += distance
        
        # Normalize by the number of pairs
        n_pairs = k * (k - 1) / 2
        avg_distances = total_distances / n_pairs
        
        return avg_distances
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        device = y_pred.device
        
        if y_pred.ndim > 1 and y_pred.size(1) > 1:
            # multiclass
            y_pred = y_pred.topk(1, 1)[1]
        
        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        
        batch_size = y_pred_2d.shape[0]
        
        # Get the top-k items for each user/query
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        
        # Compute diversity based on pairwise distances
        batch_diversity = self._compute_pairwise_distance(top_k_indices)
        
        self.sum_diversity += batch_diversity.sum().item()
        self.count += batch_size
        
        return np.array(self.sum_diversity / max(self.count, 1))


class NDCG_at_k(Metric):
    r"""
    Normalized Discounted Cumulative Gain (NDCG) at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. 
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.
    eps: float, default = 1e-8
        Small value to avoid division by zero.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import NDCG_at_k
    >>>
    >>> ndcg = NDCG_at_k(k=10)
    >>> y_pred = torch.rand(100, 5)
    >>> y_true = torch.randint(2, (100,))
    >>> score = ndcg(y_pred, y_true)
    """

    def __init__(self, n_cols: int = 10, k: Optional[int] = None, eps: float = 1e-8):
        super(NDCG_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self.eps = eps
        self._name = f"ndcg@{k}"
        self.reset()

    def reset(self):
        self.sum_ndcg = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        # NDGC@k is supposed to be used when the output reflects interest
        # scores, i.e, could be used in a regression or a multiclass problem.
        # If regression y_pred will be a float tensor, if multiclass, y_pred
        # will be a float tensor with the output of a softmax activation
        # function and we need to turn it into a 1D tensor with the class.
        # Finally, for binary problems, please use BinaryNDCG_at_k
        device = y_pred.device

        if y_pred.ndim > 1 and y_pred.size(1) > 1:
            # multiclass
            y_pred = y_pred.topk(1, 1)[1]

        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)

        batch_size = y_true_2d.shape[0]

        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        top_k_relevance = y_true_2d.gather(1, top_k_indices)
        discounts = 1.0 / torch.log2(
            torch.arange(2, top_k_relevance.shape[1] + 2, device=device)
        )

        dcg = (torch.pow(2, top_k_relevance) - 1) * discounts.unsqueeze(0)
        dcg = dcg.sum(dim=1)

        sorted_relevance, _ = torch.sort(y_true_2d, dim=1, descending=True)
        ideal_relevance = sorted_relevance[:, : self.k]

        idcg = (torch.pow(2, ideal_relevance) - 1) * discounts[
            : ideal_relevance.shape[1]
        ].unsqueeze(0)

        idcg = idcg.sum(dim=1)
        ndcg = dcg / (idcg + self.eps)

        self.sum_ndcg += ndcg.sum().item()
        self.count += batch_size

        return np.array(self.sum_ndcg / max(self.count, 1))


