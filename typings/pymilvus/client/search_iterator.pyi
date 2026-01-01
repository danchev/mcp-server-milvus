from typing import Callable

from _typeshed import Incomplete
from pymilvus.client import entity_helper as entity_helper
from pymilvus.client import utils as utils
from pymilvus.client.constants import COLLECTION_ID as COLLECTION_ID
from pymilvus.client.constants import GUARANTEE_TIMESTAMP as GUARANTEE_TIMESTAMP
from pymilvus.client.constants import ITER_SEARCH_BATCH_SIZE_KEY as ITER_SEARCH_BATCH_SIZE_KEY
from pymilvus.client.constants import ITER_SEARCH_ID_KEY as ITER_SEARCH_ID_KEY
from pymilvus.client.constants import ITER_SEARCH_LAST_BOUND_KEY as ITER_SEARCH_LAST_BOUND_KEY
from pymilvus.client.constants import ITER_SEARCH_V2_KEY as ITER_SEARCH_V2_KEY
from pymilvus.client.constants import ITERATOR_FIELD as ITERATOR_FIELD
from pymilvus.client.search_result import Hit as Hit
from pymilvus.client.search_result import Hits as Hits
from pymilvus.exceptions import ExceptionsMessage as ExceptionsMessage
from pymilvus.exceptions import ParamError as ParamError
from pymilvus.exceptions import ServerVersionIncompatibleException as ServerVersionIncompatibleException
from pymilvus.orm.connections import Connections as Connections
from pymilvus.orm.constants import MAX_BATCH_SIZE as MAX_BATCH_SIZE
from pymilvus.orm.constants import OFFSET as OFFSET
from pymilvus.orm.constants import UNLIMITED as UNLIMITED
from pymilvus.orm.iterator import SearchPage as SearchPage
from pymilvus.orm.iterator import fall_back_to_latest_session_ts as fall_back_to_latest_session_ts

logger: Incomplete

class SearchIteratorV2:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        data: list | utils.SparseMatrixInputType,
        batch_size: int = 1000,
        limit: int | None = ...,
        filter: str | None = None,
        output_fields: list[str] | None = None,
        search_params: dict | None = None,
        timeout: float | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        round_decimal: int | None = -1,
        external_filter_func: Callable[[Hits], Hits | list[Hit]] | None = None,
        **kwargs,
    ) -> None: ...
    def next(self): ...
    def close(self) -> None: ...
