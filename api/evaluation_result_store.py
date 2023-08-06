from abc import ABC, abstractmethod
from functools import lru_cache

from .dto import EvaluationResponse

class EvaluationStore:

    @abstractmethod
    def get(self, token:str) -> EvaluationResponse:
        pass

    @abstractmethod
    def write(self, token: str, eval_response: EvaluationResponse) -> bool:
        pass

    @abstractmethod
    def delete(self, token: str) -> bool:
        pass

class InMemoryEvalStore(EvaluationStore):
    """
    The simplest InMemory store of evaluation result
    """

    def __init__(self):
        self.token2eval = {}

    def get(self, token: str) -> EvaluationResponse:
        if token not in self.token2eval:
            raise ValueError("Token not found")

        return self.token2eval[token]

    def write(self, token: str, eval_response: EvaluationResponse) -> bool:
        if token in self.token2eval:
            raise False
        else:
            self.token2eval[token] = eval_response
            return True

    def delete(self, token: str) -> bool:
        if token not in self.token2eval:
            return False
        else:
            del self.token2eval[token]
            return True

@lru_cache(maxsize=1)
def load_eval_store() -> EvaluationStore:
    return InMemoryEvalStore()
