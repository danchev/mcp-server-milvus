from _typeshed import Incomplete
from pymilvus.orm.connections import connections as connections

logger: Incomplete

def create_connection(uri: str, token: str = '', db_name: str = '', use_async: bool = False, *, user: str = '', password: str = '', **kwargs) -> str: ...
