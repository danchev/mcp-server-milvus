from .grpc_gen import common_pb2 as common_pb2
from enum import IntEnum

class ErrorCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    RATE_LIMIT = 8
    FORCE_DENY = 9
    COLLECTION_NOT_FOUND = 100
    INDEX_NOT_FOUND = 700

class MilvusException(Exception):
    def __init__(self, code: int = ..., message: str = '', compatible_code: int = ...) -> None: ...
    @property
    def code(self): ...
    @property
    def message(self): ...
    @property
    def compatible_code(self): ...

class ParamError(MilvusException): ...
class ConnectError(MilvusException): ...
class MilvusUnavailableException(MilvusException): ...
class CollectionNotExistException(MilvusException): ...
class DescribeCollectionException(MilvusException): ...
class PartitionAlreadyExistException(MilvusException): ...
class IndexNotExistException(MilvusException): ...
class AmbiguousIndexName(MilvusException): ...
class CannotInferSchemaException(MilvusException): ...
class SchemaNotReadyException(MilvusException): ...
class DataTypeNotMatchException(MilvusException): ...
class DataTypeNotSupportException(MilvusException): ...
class DataNotMatchException(MilvusException): ...
class ConnectionNotExistException(MilvusException): ...
class ConnectionConfigException(MilvusException): ...
class PrimaryKeyException(MilvusException): ...
class PartitionKeyException(MilvusException): ...
class ClusteringKeyException(MilvusException): ...
class FieldsTypeException(MilvusException): ...
class FunctionsTypeException(MilvusException): ...
class FieldTypeException(MilvusException): ...
class AutoIDException(MilvusException): ...
class InvalidConsistencyLevel(MilvusException): ...
class ServerVersionIncompatibleException(MilvusException): ...

class ExceptionsMessage:
    NoHostPort: str
    HostType: str
    PortType: str
    ConnDiffConf: str
    AliasType: str
    ConnLackConf: str
    ConnectFirst: str
    CollectionNotExistNoSchema: str
    NoSchema: str
    EmptySchema: str
    SchemaType: str
    SchemaInconsistent: str
    AutoIDWithData: str
    AutoIDType: str
    NumPartitionsType: str
    AutoIDInconsistent: str
    AutoIDIllegalRanges: str
    ConsistencyLevelInconsistent: str
    AutoIDOnlyOnPK: str
    AutoIDFieldType: str
    NumberRowsInvalid: str
    FieldsNumInconsistent: str
    NoVector: str
    NoneDataFrame: str
    DataFrameType: str
    NoPrimaryKey: str
    PrimaryKeyNotExist: str
    PrimaryKeyOnlyOne: str
    PartitionKeyOnlyOne: str
    PrimaryKeyType: str
    PartitionKeyType: str
    PartitionKeyNotPrimary: str
    IsPrimaryType: str
    PrimaryFieldType: str
    PartitionKeyFieldType: str
    PartitionKeyFieldNotExist: str
    IsPartitionKeyType: str
    DataTypeInconsistent: str
    FieldDataInconsistent: str
    DataTypeNotSupport: str
    DataLengthsInconsistent: str
    DataFrameInvalid: str
    NdArrayNotSupport: str
    TypeOfDataAndSchemaInconsistent: str
    PartitionAlreadyExist: str
    IndexNotExist: str
    CollectionType: str
    FieldsType: str
    FunctionsType: str
    FunctionIncorrectInputOutputType: str
    FunctionInvalidOutputField: str
    FunctionDuplicateInputs: str
    FunctionDuplicateOutputs: str
    FunctionCommonInputOutput: str
    BM25FunctionIncorrectInputOutputCount: str
    TextEmbeddingFunctionIncorrectInputOutputCount: str
    TextEmbeddingFunctionIncorrectInputFieldType: str
    TextEmbeddingFunctionIncorrectOutputFieldType: str
    BM25FunctionIncorrectInputFieldType: str
    BM25FunctionIncorrectOutputFieldType: str
    FunctionMissingInputField: str
    FunctionMissingOutputField: str
    UnknownFunctionType: str
    FunctionIncorrectType: str
    FieldType: str
    FieldDtype: str
    ExprType: str
    EnvConfigErr: str
    AmbiguousIndexName: str
    InsertUnexpectedField: str
    InsertUnexpectedFunctionOutputField: str
    InsertMissedField: str
    UpsertAutoIDTrue: str
    AmbiguousDeleteFilterParam: str
    AmbiguousQueryFilterParam: str
    JSONKeyMustBeStr: str
    ClusteringKeyType: str
    ClusteringKeyFieldNotExist: str
    ClusteringKeyOnlyOne: str
    IsClusteringKeyType: str
    ClusteringKeyFieldType: str
    UpsertPrimaryKeyEmpty: str
    DefaultValueInvalid: str
    SearchIteratorV2FallbackWarning: str
