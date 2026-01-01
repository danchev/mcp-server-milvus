import pandas as pd
from _typeshed import Incomplete
from pymilvus.client import utils as utils
from pymilvus.client.abstract import BaseRanker as BaseRanker
from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL as DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.search_result import SearchResult as SearchResult
from pymilvus.client.types import CompactionPlans as CompactionPlans
from pymilvus.client.types import CompactionState as CompactionState
from pymilvus.client.types import Replica as Replica
from pymilvus.client.types import cmp_consistency_level as cmp_consistency_level
from pymilvus.client.types import get_consistency_level as get_consistency_level
from pymilvus.exceptions import AutoIDException as AutoIDException
from pymilvus.exceptions import DataTypeNotMatchException as DataTypeNotMatchException
from pymilvus.exceptions import DataTypeNotSupportException as DataTypeNotSupportException
from pymilvus.exceptions import ExceptionsMessage as ExceptionsMessage
from pymilvus.exceptions import IndexNotExistException as IndexNotExistException
from pymilvus.exceptions import PartitionAlreadyExistException as PartitionAlreadyExistException
from pymilvus.exceptions import SchemaNotReadyException as SchemaNotReadyException
from pymilvus.grpc_gen import schema_pb2 as schema_pb2
from pymilvus.settings import Config as Config

from .connections import connections as connections
from .constants import UNLIMITED as UNLIMITED
from .future import MutationFuture as MutationFuture
from .future import SearchFuture as SearchFuture
from .index import Index as Index
from .iterator import QueryIterator as QueryIterator
from .iterator import SearchIterator as SearchIterator
from .mutation import MutationResult as MutationResult
from .partition import Partition as Partition
from .prepare import Prepare as Prepare
from .schema import CollectionSchema as CollectionSchema
from .schema import FieldSchema as FieldSchema
from .schema import Function as Function
from .schema import check_insert_schema as check_insert_schema
from .schema import check_schema as check_schema
from .schema import check_upsert_schema as check_upsert_schema
from .schema import construct_fields_from_dataframe as construct_fields_from_dataframe
from .schema import is_row_based as is_row_based
from .schema import is_valid_insert_data as is_valid_insert_data
from .types import DataType as DataType

class Collection:
    def __init__(self, name: str, schema: CollectionSchema | None = None, using: str = "default", **kwargs) -> None: ...
    @classmethod
    def construct_from_dataframe(cls, name: str, dataframe: pd.DataFrame, **kwargs): ...
    @property
    def schema(self) -> CollectionSchema: ...
    @property
    def aliases(self) -> list: ...
    @property
    def description(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def num_shards(self) -> int: ...
    @property
    def num_entities(self) -> int: ...
    @property
    def primary_field(self) -> FieldSchema: ...
    def flush(self, timeout: float | None = None, **kwargs): ...
    def drop(self, timeout: float | None = None, **kwargs): ...
    def set_properties(self, properties: dict, timeout: float | None = None, **kwargs): ...
    def load(
        self,
        partition_names: list | None = None,
        replica_number: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def release(self, timeout: float | None = None, **kwargs): ...
    def insert(
        self,
        data: list | pd.DataFrame | dict | utils.SparseMatrixInputType,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> MutationResult: ...
    def delete(self, expr: str, partition_name: str | None = None, timeout: float | None = None, **kwargs): ...
    def upsert(
        self,
        data: list | pd.DataFrame | dict | utils.SparseMatrixInputType,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> MutationResult: ...
    def search(
        self,
        data: list | utils.SparseMatrixInputType,
        anns_field: str,
        param: dict,
        limit: int,
        expr: str | None = None,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        round_decimal: int = -1,
        ranker: Function | None = None,
        **kwargs,
    ): ...
    def hybrid_search(
        self,
        reqs: list,
        rerank: BaseRanker,
        limit: int,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        round_decimal: int = -1,
        ranker: Function | None = None,
        **kwargs,
    ): ...
    def search_iterator(
        self,
        data: list | utils.SparseMatrixInputType,
        anns_field: str,
        param: dict,
        batch_size: int | None = 1000,
        limit: int | None = ...,
        expr: str | None = None,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        round_decimal: int = -1,
        **kwargs,
    ): ...
    def query(
        self,
        expr: str,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def query_iterator(
        self,
        batch_size: int | None = 1000,
        limit: int | None = ...,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    @property
    def partitions(self) -> list[Partition]: ...
    def partition(self, partition_name: str, **kwargs) -> Partition: ...
    def create_partition(self, partition_name: str, description: str = "", **kwargs) -> Partition: ...
    def has_partition(self, partition_name: str, timeout: float | None = None, **kwargs) -> bool: ...
    def drop_partition(self, partition_name: str, timeout: float | None = None, **kwargs): ...
    @property
    def indexes(self) -> list[Index]: ...
    def index(self, **kwargs) -> Index: ...
    def create_index(
        self, field_name: str, index_params: dict | None = None, timeout: float | None = None, **kwargs
    ): ...
    def alter_index(self, index_name: str, extra_params: dict, timeout: float | None = None): ...
    def has_index(self, timeout: float | None = None, **kwargs) -> bool: ...
    def drop_index(self, timeout: float | None = None, **kwargs): ...
    clustering_compaction_id: Incomplete
    compaction_id: Incomplete
    def compact(self, is_clustering: bool | None = False, timeout: float | None = None, **kwargs): ...
    def get_compaction_state(
        self, timeout: float | None = None, is_clustering: bool | None = False, **kwargs
    ) -> CompactionState: ...
    def wait_for_compaction_completed(
        self, timeout: float | None = None, is_clustering: bool | None = False, **kwargs
    ) -> CompactionState: ...
    def get_compaction_plans(
        self, timeout: float | None = None, is_clustering: bool | None = False, **kwargs
    ) -> CompactionPlans: ...
    def get_replicas(self, timeout: float | None = None, **kwargs) -> Replica: ...
    def describe(self, timeout: float | None = None): ...
