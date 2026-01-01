from typing import Any, Callable, TypeVar

from _typeshed import Incomplete
from pymilvus.client import entity_helper as entity_helper
from pymilvus.client import utils as utils
from pymilvus.client.abstract import LoopBase as LoopBase
from pymilvus.client.search_result import Hits as Hits
from pymilvus.exceptions import MilvusException as MilvusException
from pymilvus.exceptions import ParamError as ParamError
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

from .connections import Connections as Connections
from .constants import BATCH_SIZE as BATCH_SIZE
from .constants import CALC_DIST_BM25 as CALC_DIST_BM25
from .constants import CALC_DIST_COSINE as CALC_DIST_COSINE
from .constants import CALC_DIST_HAMMING as CALC_DIST_HAMMING
from .constants import CALC_DIST_IP as CALC_DIST_IP
from .constants import CALC_DIST_JACCARD as CALC_DIST_JACCARD
from .constants import CALC_DIST_L2 as CALC_DIST_L2
from .constants import CALC_DIST_TANIMOTO as CALC_DIST_TANIMOTO
from .constants import COLLECTION_ID as COLLECTION_ID
from .constants import DEFAULT_SEARCH_EXTENSION_RATE as DEFAULT_SEARCH_EXTENSION_RATE
from .constants import EF as EF
from .constants import FIELDS as FIELDS
from .constants import GUARANTEE_TIMESTAMP as GUARANTEE_TIMESTAMP
from .constants import INT64_MAX as INT64_MAX
from .constants import IS_PRIMARY as IS_PRIMARY
from .constants import ITERATOR_FIELD as ITERATOR_FIELD
from .constants import ITERATOR_SESSION_CP_FILE as ITERATOR_SESSION_CP_FILE
from .constants import ITERATOR_SESSION_TS_FIELD as ITERATOR_SESSION_TS_FIELD
from .constants import MAX_BATCH_SIZE as MAX_BATCH_SIZE
from .constants import MAX_FILTERED_IDS_COUNT_ITERATION as MAX_FILTERED_IDS_COUNT_ITERATION
from .constants import MAX_TRY_TIME as MAX_TRY_TIME
from .constants import METRIC_TYPE as METRIC_TYPE
from .constants import MILVUS_LIMIT as MILVUS_LIMIT
from .constants import OFFSET as OFFSET
from .constants import PARAMS as PARAMS
from .constants import RADIUS as RADIUS
from .constants import RANGE_FILTER as RANGE_FILTER
from .constants import REDUCE_STOP_FOR_BEST as REDUCE_STOP_FOR_BEST
from .constants import UNLIMITED as UNLIMITED
from .schema import CollectionSchema as CollectionSchema
from .types import DataType as DataType
from .utility import mkts_from_datetime as mkts_from_datetime

log: Incomplete
QueryIterator = TypeVar("QueryIterator")
SearchIterator = TypeVar("SearchIterator")

def fall_back_to_latest_session_ts(): ...
def assert_info(condition: bool, message: str): ...
def io_operation(io_func: Callable[[Any], None], message: str): ...
def extend_batch_size(batch_size: int, next_param: dict, to_extend_batch_size: bool) -> int: ...
def check_set_flag(obj: Any, flag_name: str, kwargs: dict[str, Any], key: str): ...

class QueryIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        batch_size: int | None = 1000,
        limit: int | None = -1,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        schema: CollectionSchema | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> None: ...
    def get_cursor(self) -> milvus_types.QueryCursor: ...
    def next(self): ...
    def close(self) -> None: ...

def metrics_positive_related(metrics: str) -> bool: ...

class SearchPage(LoopBase):
    def __init__(self, res: Hits, session_ts: int | None = 0) -> None: ...
    def get_session_ts(self): ...
    def get_res(self): ...
    def __len__(self) -> int: ...
    def get__item(self, idx: Any): ...
    def merge(self, others: list[Hits]): ...
    def ids(self): ...
    def distances(self): ...

class SearchIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        data: list | utils.SparseMatrixInputType,
        ann_field: str,
        param: dict,
        batch_size: int | None = 1000,
        limit: int | None = ...,
        expr: str | None = None,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        round_decimal: int = -1,
        schema: CollectionSchema | None = None,
        **kwargs,
    ) -> None: ...
    def next(self): ...
    def close(self) -> None: ...

class IteratorCache:
    def __init__(self) -> None: ...
    def cache(self, result: Any, cache_id: int): ...
    def fetch_cache(self, cache_id: int): ...
    def release_cache(self, cache_id: int): ...

NO_CACHE_ID: int
iterator_cache: Incomplete
