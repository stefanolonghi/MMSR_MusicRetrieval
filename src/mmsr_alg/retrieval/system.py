from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from ..catalog import Catalog

@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    algo: str
    k: int
    ranked_ids: List[str]
    scores: Optional[List[float]]

AlgoFn = Callable[[Catalog, int, int, Optional[int]], RetrievalResult]

class RetrievalSystem:
    def __init__(self, catalog: Catalog, algorithms: Dict[str, AlgoFn]):
        self.catalog = catalog
        self.algorithms = algorithms

    def retrieve(self, query_id: str, k: int, algo: str, seed: Optional[int] = None) -> RetrievalResult:
        qidx = self.catalog.id_to_idx[query_id]
        return self.algorithms[algo](self.catalog, qidx, k, seed)
