import pandas as pd
from pymilvus.client.types import DataType as DataType
from pymilvus.exceptions import DataNotMatchException as DataNotMatchException
from pymilvus.exceptions import DataTypeNotSupportException as DataTypeNotSupportException
from pymilvus.exceptions import ExceptionsMessage as ExceptionsMessage
from pymilvus.exceptions import ParamError as ParamError

from .schema import CollectionSchema as CollectionSchema

class Prepare:
    @classmethod
    def prepare_data(
        cls, data: list | tuple | pd.DataFrame, schema: CollectionSchema, is_insert: bool = True
    ) -> list: ...
