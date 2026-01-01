from typing import Any, Callable, Iterable, Mapping

import grpc
from _typeshed import Incomplete
from pymilvus.decorators import ignore_unimplemented as ignore_unimplemented
from pymilvus.decorators import retry_on_rpc_failure as retry_on_rpc_failure
from pymilvus.decorators import upgrade_reminder as upgrade_reminder
from pymilvus.exceptions import AmbiguousIndexName as AmbiguousIndexName
from pymilvus.exceptions import DataNotMatchException as DataNotMatchException
from pymilvus.exceptions import DescribeCollectionException as DescribeCollectionException
from pymilvus.exceptions import ErrorCode as ErrorCode
from pymilvus.exceptions import ExceptionsMessage as ExceptionsMessage
from pymilvus.exceptions import MilvusException as MilvusException
from pymilvus.exceptions import ParamError as ParamError
from pymilvus.grpc_gen import common_pb2 as common_pb2
from pymilvus.grpc_gen import milvus_pb2_grpc as milvus_pb2_grpc
from pymilvus.orm.schema import Function as Function
from pymilvus.settings import Config as Config

from . import entity_helper as entity_helper
from . import interceptor as interceptor
from . import ts_utils as ts_utils
from . import utils as utils
from .abstract import AnnSearchRequest as AnnSearchRequest
from .abstract import BaseRanker as BaseRanker
from .abstract import CollectionSchema as CollectionSchema
from .abstract import FieldSchema as FieldSchema
from .abstract import MutationResult as MutationResult
from .asynch import CreateIndexFuture as CreateIndexFuture
from .asynch import FlushFuture as FlushFuture
from .asynch import MutationFuture as MutationFuture
from .asynch import SearchFuture as SearchFuture
from .check import check_pass_param as check_pass_param
from .check import is_legal_host as is_legal_host
from .check import is_legal_port as is_legal_port
from .constants import ITERATOR_SESSION_TS_FIELD as ITERATOR_SESSION_TS_FIELD
from .prepare import Prepare as Prepare
from .search_result import SearchResult as SearchResult
from .types import AnalyzeResult as AnalyzeResult
from .types import BulkInsertState as BulkInsertState
from .types import CompactionPlans as CompactionPlans
from .types import CompactionState as CompactionState
from .types import DatabaseInfo as DatabaseInfo
from .types import GrantInfo as GrantInfo
from .types import Group as Group
from .types import HybridExtraList as HybridExtraList
from .types import IndexState as IndexState
from .types import LoadState as LoadState
from .types import Plan as Plan
from .types import PrivilegeGroupInfo as PrivilegeGroupInfo
from .types import Replica as Replica
from .types import ReplicaInfo as ReplicaInfo
from .types import ResourceGroupConfig as ResourceGroupConfig
from .types import ResourceGroupInfo as ResourceGroupInfo
from .types import RoleInfo as RoleInfo
from .types import Shard as Shard
from .types import State as State
from .types import Status as Status
from .types import UserInfo as UserInfo
from .types import get_cost_extra as get_cost_extra
from .utils import check_invalid_binary_vector as check_invalid_binary_vector
from .utils import check_status as check_status
from .utils import get_server_type as get_server_type
from .utils import is_successful as is_successful
from .utils import len_of as len_of

class GrpcHandler:
    callbacks: Incomplete
    schema_cache: Incomplete
    def __init__(
        self, uri: str = ..., host: str = "", port: str = "", channel: grpc.Channel | None = None, **kwargs
    ) -> None: ...
    def register_state_change_callback(self, callback: Callable): ...
    def deregister_state_change_callbacks(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object): ...
    def close(self) -> None: ...
    def reset_db_name(self, db_name: str): ...
    def set_onetime_loglevel(self, log_level: str): ...
    @property
    def server_address(self): ...
    def get_server_type(self): ...
    def reset_password(
        self, user: str, old_password: str, new_password: str, timeout: float | None = None, **kwargs
    ): ...
    def create_collection(
        self,
        collection_name: str,
        fields: CollectionSchema | dict[str, Iterable],
        timeout: float | None = None,
        **kwargs,
    ): ...
    def drop_collection(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def add_collection_field(
        self, collection_name: str, field_schema: FieldSchema, timeout: float | None = None, **kwargs
    ): ...
    def alter_collection_properties(
        self, collection_name: str, properties: list, timeout: float | None = None, **kwargs
    ): ...
    def alter_collection_field_properties(
        self,
        collection_name: str,
        field_name: str,
        field_params: dict[str, Any],
        timeout: float | None = None,
        **kwargs,
    ): ...
    def drop_collection_properties(
        self, collection_name: str, property_keys: list[str], timeout: float | None = None, **kwargs
    ): ...
    def has_collection(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def describe_collection(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def list_collections(self, timeout: float | None = None, **kwargs): ...
    def rename_collections(
        self, old_name: str, new_name: str, new_db_name: str = "", timeout: float | None = None, **kwargs
    ): ...
    def create_partition(self, collection_name: str, partition_name: str, timeout: float | None = None, **kwargs): ...
    def drop_partition(self, collection_name: str, partition_name: str, timeout: float | None = None, **kwargs): ...
    def has_partition(self, collection_name: str, partition_name: str, timeout: float | None = None, **kwargs): ...
    def list_partitions(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: float | None = None, **kwargs
    ): ...
    def insert_rows(
        self,
        collection_name: str,
        entities: dict | list[dict],
        partition_name: str | None = None,
        schema: dict | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def update_schema(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def batch_insert(
        self,
        collection_name: str,
        entities: list,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def delete(
        self,
        collection_name: str,
        expression: str,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def upsert(
        self,
        collection_name: str,
        entities: list,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def upsert_rows(
        self,
        collection_name: str,
        entities: list,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def search(
        self,
        collection_name: str,
        data: list[list[float]] | utils.SparseMatrixInputType,
        anns_field: str,
        param: dict,
        limit: int,
        expression: str | None = None,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        round_decimal: int = -1,
        timeout: float | None = None,
        ranker: Function | None = None,
        **kwargs,
    ): ...
    def hybrid_search(
        self,
        collection_name: str,
        reqs: list[AnnSearchRequest],
        rerank: BaseRanker | Function,
        limit: int,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        round_decimal: int = -1,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def get_query_segment_info(self, collection_name: str, timeout: float = 30, **kwargs): ...
    def create_alias(self, collection_name: str, alias: str, timeout: float | None = None, **kwargs): ...
    def drop_alias(self, alias: str, timeout: float | None = None, **kwargs): ...
    def alter_alias(self, collection_name: str, alias: str, timeout: float | None = None, **kwargs): ...
    def describe_alias(self, alias: str, timeout: float | None = None, **kwargs): ...
    def list_aliases(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def create_index(
        self, collection_name: str, field_name: str, params: dict, timeout: float | None = None, **kwargs
    ): ...
    def alter_index_properties(
        self, collection_name: str, index_name: str, properties: dict, timeout: float | None = None, **kwargs
    ): ...
    def drop_index_properties(
        self, collection_name: str, index_name: str, property_keys: list[str], timeout: float | None = None, **kwargs
    ): ...
    def list_indexes(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: float | None = None,
        timestamp: int | None = None,
        **kwargs,
    ): ...
    def get_index_build_progress(
        self, collection_name: str, index_name: str, timeout: float | None = None, **kwargs
    ): ...
    def get_index_state(
        self,
        collection_name: str,
        index_name: str,
        timeout: float | None = None,
        timestamp: int | None = None,
        **kwargs,
    ): ...
    def wait_for_creating_index(
        self, collection_name: str, index_name: str, timeout: float | None = None, **kwargs
    ): ...
    def load_collection(
        self, collection_name: str, replica_number: int | None = None, timeout: float | None = None, **kwargs
    ): ...
    def load_collection_progress(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def wait_for_loading_collection(
        self, collection_name: str, timeout: float | None = None, is_refresh: bool = False, **kwargs
    ): ...
    def release_collection(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def load_partitions(
        self,
        collection_name: str,
        partition_names: list[str],
        replica_number: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def wait_for_loading_partitions(
        self,
        collection_name: str,
        partition_names: list[str],
        timeout: float | None = None,
        is_refresh: bool = False,
        **kwargs,
    ): ...
    def get_loading_progress(
        self,
        collection_name: str,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        is_refresh: bool = False,
        **kwargs,
    ): ...
    def create_database(self, db_name: str, properties: dict | None = None, timeout: float | None = None, **kwargs): ...
    def drop_database(self, db_name: str, timeout: float | None = None, **kwargs): ...
    def list_database(self, timeout: float | None = None, **kwargs): ...
    def alter_database(self, db_name: str, properties: dict, timeout: float | None = None, **kwargs): ...
    def drop_database_properties(
        self, db_name: str, property_keys: list[str], timeout: float | None = None, **kwargs
    ): ...
    def describe_database(self, db_name: str, timeout: float | None = None, **kwargs): ...
    def get_load_state(
        self, collection_name: str, partition_names: list[str] | None = None, timeout: float | None = None, **kwargs
    ): ...
    def load_partitions_progress(
        self, collection_name: str, partition_names: list[str], timeout: float | None = None, **kwargs
    ): ...
    def release_partitions(
        self, collection_name: str, partition_names: list[str], timeout: float | None = None, **kwargs
    ): ...
    def get_collection_stats(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def get_flush_state(
        self, segment_ids: list[int], collection_name: str, flush_ts: int, timeout: float | None = None, **kwargs
    ): ...
    def get_persistent_segment_infos(self, collection_name: str, timeout: float | None = None, **kwargs): ...
    def flush(self, collection_names: list, timeout: float | None = None, **kwargs): ...
    def drop_index(
        self, collection_name: str, field_name: str, index_name: str, timeout: float | None = None, **kwargs
    ): ...
    def dummy(self, request_type: Any, timeout: float | None = None, **kwargs): ...
    def fake_register_link(self, timeout: float | None = None, **kwargs): ...
    def get(
        self,
        collection_name: str,
        ids: list[int],
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        strict_float32: bool = False,
        **kwargs,
    ): ...
    def load_balance(
        self,
        collection_name: str,
        src_node_id: int,
        dst_node_ids: list[int],
        sealed_segment_ids: list[int],
        timeout: float | None = None,
        **kwargs,
    ): ...
    def compact(
        self, collection_name: str, is_clustering: bool | None = False, timeout: float | None = None, **kwargs
    ) -> int: ...
    def get_compaction_state(self, compaction_id: int, timeout: float | None = None, **kwargs) -> CompactionState: ...
    def wait_for_compaction_completed(self, compaction_id: int, timeout: float | None = None, **kwargs): ...
    def get_compaction_plans(self, compaction_id: int, timeout: float | None = None, **kwargs) -> CompactionPlans: ...
    def get_replicas(self, collection_name: str, timeout: float | None = None, **kwargs) -> Replica: ...
    def describe_replica(self, collection_name: str, timeout: float | None = None, **kwargs) -> list[ReplicaInfo]: ...
    def do_bulk_insert(
        self, collection_name: str, partition_name: str, files: list[str], timeout: float | None = None, **kwargs
    ) -> int: ...
    def get_bulk_insert_state(self, task_id: int, timeout: float | None = None, **kwargs) -> BulkInsertState: ...
    def list_bulk_insert_tasks(
        self, limit: int, collection_name: str, timeout: float | None = None, **kwargs
    ) -> list: ...
    def create_user(self, user: str, password: str, timeout: float | None = None, **kwargs): ...
    def update_password(
        self, user: str, old_password: str, new_password: str, timeout: float | None = None, **kwargs
    ): ...
    def delete_user(self, user: str, timeout: float | None = None, **kwargs): ...
    def list_usernames(self, timeout: float | None = None, **kwargs): ...
    def create_role(self, role_name: str, timeout: float | None = None, **kwargs): ...
    def drop_role(self, role_name: str, force_drop: bool = False, timeout: float | None = None, **kwargs): ...
    def add_user_to_role(self, username: str, role_name: str, timeout: float | None = None, **kwargs): ...
    def remove_user_from_role(self, username: str, role_name: str, timeout: float | None = None, **kwargs): ...
    def select_one_role(self, role_name: str, include_user_info: bool, timeout: float | None = None, **kwargs): ...
    def select_all_role(self, include_user_info: bool, timeout: float | None = None, **kwargs): ...
    def select_one_user(self, username: str, include_role_info: bool, timeout: float | None = None, **kwargs): ...
    def select_all_user(self, include_role_info: bool, timeout: float | None = None, **kwargs): ...
    def grant_privilege(
        self,
        role_name: str,
        object: str,
        object_name: str,
        privilege: str,
        db_name: str,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def revoke_privilege(
        self,
        role_name: str,
        object: str,
        object_name: str,
        privilege: str,
        db_name: str,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def grant_privilege_v2(
        self,
        role_name: str,
        privilege: str,
        collection_name: str,
        db_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def revoke_privilege_v2(
        self,
        role_name: str,
        privilege: str,
        collection_name: str,
        db_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
    def select_grant_for_one_role(self, role_name: str, db_name: str, timeout: float | None = None, **kwargs): ...
    def select_grant_for_role_and_object(
        self, role_name: str, object: str, object_name: str, db_name: str, timeout: float | None = None, **kwargs
    ): ...
    def get_server_version(self, timeout: float | None = None, **kwargs) -> str: ...
    def create_resource_group(self, name: str, timeout: float | None = None, **kwargs): ...
    def update_resource_groups(
        self, configs: Mapping[str, ResourceGroupConfig], timeout: float | None = None, **kwargs
    ): ...
    def drop_resource_group(self, name: str, timeout: float | None = None, **kwargs): ...
    def list_resource_groups(self, timeout: float | None = None, **kwargs): ...
    def describe_resource_group(self, name: str, timeout: float | None = None, **kwargs) -> ResourceGroupInfo: ...
    def transfer_node(self, source: str, target: str, num_node: int, timeout: float | None = None, **kwargs): ...
    def transfer_replica(
        self, source: str, target: str, collection_name: str, num_replica: int, timeout: float | None = None, **kwargs
    ): ...
    def get_flush_all_state(self, flush_all_ts: int, timeout: float | None = None, **kwargs): ...
    def flush_all(self, timeout: float | None = None, **kwargs): ...
    def alloc_timestamp(self, timeout: float | None = None, **kwargs) -> int: ...
    def create_privilege_group(self, privilege_group: str, timeout: float | None = None, **kwargs): ...
    def drop_privilege_group(self, privilege_group: str, timeout: float | None = None, **kwargs): ...
    def list_privilege_groups(self, timeout: float | None = None, **kwargs): ...
    def add_privileges_to_group(
        self, privilege_group: str, privileges: list[str], timeout: float | None = None, **kwargs
    ): ...
    def remove_privileges_from_group(
        self, privilege_group: str, privileges: list[str], timeout: float | None = None, **kwargs
    ): ...
    def run_analyzer(
        self,
        texts: str | list[str],
        analyzer_params: str | dict | None = None,
        with_hash: bool = False,
        with_detail: bool = False,
        collection_name: str | None = None,
        field_name: str | None = None,
        analyzer_names: str | list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ): ...
