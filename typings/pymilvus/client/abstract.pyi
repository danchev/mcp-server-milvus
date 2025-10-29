import abc
from . import utils as utils
from .constants import DEFAULT_CONSISTENCY_LEVEL as DEFAULT_CONSISTENCY_LEVEL, RANKER_TYPE_RRF as RANKER_TYPE_RRF, RANKER_TYPE_WEIGHTED as RANKER_TYPE_WEIGHTED
from .search_result import Hit as Hit, Hits as Hits, SearchResult as SearchResult
from .types import DataType as DataType, FunctionType as FunctionType
from _typeshed import Incomplete
from pymilvus.exceptions import DataTypeNotMatchException as DataTypeNotMatchException, ExceptionsMessage as ExceptionsMessage
from pymilvus.settings import Config as Config
from typing import Any

logger: Incomplete

class FieldSchema:
    field_id: int
    name: Incomplete
    is_primary: bool
    description: Incomplete
    auto_id: bool
    type: Incomplete
    indexes: Incomplete
    params: Incomplete
    is_partition_key: bool
    is_dynamic: bool
    nullable: bool
    default_value: Incomplete
    is_function_output: bool
    element_type: Incomplete
    is_clustering_key: bool
    def __init__(self, raw: Any) -> None: ...
    def dict(self): ...

class FunctionSchema:
    name: Incomplete
    description: Incomplete
    type: Incomplete
    params: Incomplete
    input_field_names: Incomplete
    input_field_ids: Incomplete
    output_field_names: Incomplete
    output_field_ids: Incomplete
    id: int
    def __init__(self, raw: Any) -> None: ...
    def dict(self): ...

class CollectionSchema:
    collection_name: Incomplete
    description: Incomplete
    params: Incomplete
    fields: Incomplete
    functions: Incomplete
    statistics: Incomplete
    auto_id: bool
    aliases: Incomplete
    collection_id: int
    consistency_level: Incomplete
    properties: Incomplete
    num_shards: int
    num_partitions: int
    enable_dynamic_field: bool
    created_timestamp: int
    update_timestamp: int
    def __init__(self, raw: Any) -> None: ...
    def dict(self): ...

class MutationResult:
    def __init__(self, raw: Any) -> None: ...
    @property
    def primary_keys(self): ...
    @property
    def insert_count(self): ...
    @property
    def delete_count(self): ...
    @property
    def upsert_count(self): ...
    @property
    def timestamp(self): ...
    @property
    def succ_count(self): ...
    @property
    def err_count(self): ...
    @property
    def succ_index(self): ...
    @property
    def err_index(self): ...
    @property
    def cost(self): ...

class BaseRanker:
    def __int__(self) -> int: ...
    def dict(self): ...

class RRFRanker(BaseRanker):
    def __init__(self, k: int = 60) -> None: ...
    def dict(self): ...

class WeightedRanker(BaseRanker):
    def __init__(self, *nums, norm_score: bool = True) -> None: ...
    def dict(self): ...

class AnnSearchRequest:
    def __init__(self, data: list | utils.SparseMatrixInputType, anns_field: str, param: dict, limit: int, expr: str | None = None, expr_params: dict | None = None) -> None: ...
    @property
    def data(self): ...
    @property
    def anns_field(self): ...
    @property
    def param(self): ...
    @property
    def limit(self): ...
    @property
    def expr(self): ...
    @property
    def expr_params(self): ...

class LoopBase(metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    def __iter__(self): ...
    def __getitem__(self, item: Any): ...
    def __next__(self): ...
    @abc.abstractmethod
    def get__item(self, item: Any): ...
