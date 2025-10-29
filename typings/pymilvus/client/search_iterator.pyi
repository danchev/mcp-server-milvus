from _typeshed import Incomplete
from pymilvus.client import entity_helper as entity_helper, utils as utils
from pymilvus.client.constants import COLLECTION_ID as COLLECTION_ID, GUARANTEE_TIMESTAMP as GUARANTEE_TIMESTAMP, ITERATOR_FIELD as ITERATOR_FIELD, ITER_SEARCH_BATCH_SIZE_KEY as ITER_SEARCH_BATCH_SIZE_KEY, ITER_SEARCH_ID_KEY as ITER_SEARCH_ID_KEY, ITER_SEARCH_LAST_BOUND_KEY as ITER_SEARCH_LAST_BOUND_KEY, ITER_SEARCH_V2_KEY as ITER_SEARCH_V2_KEY
from pymilvus.client.search_result import Hit as Hit, Hits as Hits
from pymilvus.exceptions import ExceptionsMessage as ExceptionsMessage, ParamError as ParamError, ServerVersionIncompatibleException as ServerVersionIncompatibleException
from pymilvus.orm.connections import Connections as Connections
from pymilvus.orm.constants import MAX_BATCH_SIZE as MAX_BATCH_SIZE, OFFSET as OFFSET, UNLIMITED as UNLIMITED
from pymilvus.orm.iterator import SearchPage as SearchPage, fall_back_to_latest_session_ts as fall_back_to_latest_session_ts
from typing import Callable

logger: Incomplete

class SearchIteratorV2:
    def __init__(self, connection: Connections, collection_name: str, data: list | utils.SparseMatrixInputType, batch_size: int = 1000, limit: int | None = ..., filter: str | None = None, output_fields: list[str] | None = None, search_params: dict | None = None, timeout: float | None = None, partition_names: list[str] | None = None, anns_field: str | None = None, round_decimal: int | None = -1, external_filter_func: Callable[[Hits], Hits | list[Hit]] | None = None, **kwargs) -> None: ...
    def next(self): ...
    def close(self) -> None: ...
