import logging
from _typeshed import Incomplete

class Config:
    LEGACY_URI: Incomplete
    MILVUS_URI: Incomplete
    MILVUS_CONN_ALIAS: Incomplete
    MILVUS_CONN_TIMEOUT: Incomplete
    DEFAULT_USING = MILVUS_CONN_ALIAS
    DEFAULT_CONNECT_TIMEOUT = MILVUS_CONN_TIMEOUT
    GRPC_PORT: str
    GRPC_ADDRESS: str
    GRPC_URI: Incomplete
    DEFAULT_HOST: str
    DEFAULT_PORT: str
    WaitTimeDurationWhenLoad: float
    MaxVarCharLengthKey: str
    MaxVarCharLength: int
    EncodeProtocol: str
    IndexName: str

COLORS: Incomplete

class ColorFulFormatColMixin:
    def format_col(self, message_str: str, level_name: str): ...

class ColorfulFormatter(logging.Formatter, ColorFulFormatColMixin):
    def format(self, record: str): ...

def init_log(log_level: str): ...
