import pandas as pd
from .schema import CollectionSchema as CollectionSchema
from pymilvus.client.types import DataType as DataType
from pymilvus.exceptions import DataNotMatchException as DataNotMatchException, DataTypeNotSupportException as DataTypeNotSupportException, ExceptionsMessage as ExceptionsMessage, ParamError as ParamError

class Prepare:
    @classmethod
    def prepare_data(cls, data: list | tuple | pd.DataFrame, schema: CollectionSchema, is_insert: bool = True) -> list: ...
