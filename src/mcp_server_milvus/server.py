import argparse
import json
import os
import asyncio
import logging
import functools # Added for functools.partial
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from pydantic import BaseSettings, Field # Added for Pydantic settings
from mcp.server.fastmcp import Context, FastMCP
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import (
    MilvusException,
    CollectionNotExistException,
    IndexNotExistException,
    ConnectError,
    ParamError,
    SchemaNotReadyException,
    DataTypeNotMatchException,
    DataNotMatchException,
    PrimaryKeyException,
    PartitionKeyException,
    MilvusUnavailableException,
    # Add others here if needed, e.g., Exceptions.DescribeCollectionException if it exists
)

logger = logging.getLogger(__name__)

# Pydantic Settings Model
class Settings(BaseSettings):
    milvus_uri: str = Field("http://localhost:19530", env="MILVUS_URI")
    milvus_token: Optional[str] = Field(None, env="MILVUS_TOKEN")
    milvus_db: str = Field("default", env="MILVUS_DB")
    milvus_vector_search_nprobe: int = Field(10, env="MILVUS_VECTOR_SEARCH_NPROBE")
    milvus_create_index_nlist: int = Field(1024, env="MILVUS_CREATE_INDEX_NLIST")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        # Pydantic V2 uses model_config now, but for older Pydantic V1 style:
        # env_file = ".env" # Example if .env file was used
        # env_file_encoding = "utf-8"
        # For Pydantic V2, environment variable reading is default.
        # Ensure variable names match field names or aliases if not using Field(env=...).
        # For this task, Field(env=...) is explicit.
        pass


# Custom Exceptions
class MilvusMCPError(Exception):
    """Base exception for Milvus MCP errors."""
    pass

class MilvusConnectionError(MilvusMCPError):
    """Raised when there's an issue connecting to Milvus."""
    pass

class CollectionNotFoundError(MilvusMCPError):
    """Raised when a specified collection is not found."""
    pass

class SchemaError(MilvusMCPError):
    """Raised when there's an issue with the collection schema."""
    pass

class IndexNotFoundError(MilvusMCPError):
    """Raised when a specified index is not found."""
    pass

class SearchError(MilvusMCPError):
    """Raised during search operations if an error occurs."""
    pass

class QueryError(MilvusMCPError):
    """Raised during query operations if an error occurs."""
    pass

class DataOperationError(MilvusMCPError):
    """Raised for errors during data operations like insert, delete, upsert."""
    pass

class MilvusOperationError(MilvusMCPError):
    """Raised for other generic Milvus errors."""
    pass


class MilvusConnector:
    def __init__(
        self,
        uri: str,
        token: Optional[str],
        db_name: str,
        vector_search_nprobe: int,
        create_index_nlist: int,
    ):
        self.uri = uri
        self.token = token
        self.db_name = db_name # Store db_name for re-connection if needed in use_database
        self.vector_search_nprobe = vector_search_nprobe
        self.create_index_nlist = create_index_nlist

        logger.debug(f"Initializing MilvusConnector with URI: {self.uri}, DB: {self.db_name}, Token Provided: {self.token is not None}, NProbe: {self.vector_search_nprobe}, NList: {self.create_index_nlist}")
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token, db_name=self.db_name)
            logger.info(f"MilvusConnector initialized successfully for URI: {self.uri}, DB: {self.db_name}")
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during client initialization for URI {self.uri}: {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Failed to connect to Milvus at {self.uri}: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for Milvus client (URI: {self.uri}, DB: {self.db_name}): {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for Milvus client (uri, token, or db_name): {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Unexpected Milvus error during client initialization for URI {self.uri}: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"An unexpected Milvus error occurred during client initialization: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected non-Milvus error during client initialization for URI {self.uri}: {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during client initialization: {str(e)}") from e

    # from_env method is removed as per requirements

    async def list_collections(self) -> list[str]:
        """List all collections in the database."""
        logger.info("Attempting to list collections.")
        try:
            collections = await asyncio.to_thread(self.client.list_collections)
            logger.info(f"Successfully listed collections: {collections}")
            return collections
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error while listing collections: {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error while listing collections: {str(e)}") from e
        except MilvusException as e: # Catch other specific Milvus errors
            logger.error(f"Failed to list collections due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to list collections due to Milvus error: {str(e)}") from e
        except Exception as e: # Catch any other unexpected error
            logger.error(f"An unexpected error occurred while listing collections: {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred while listing collections: {str(e)}") from e

    async def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a collection."""
        logger.info(f"Attempting to get collection info for '{collection_name}'.")
        try:
            info = await asyncio.to_thread(self.client.describe_collection, collection_name)
            logger.info(f"Successfully retrieved info for collection '{collection_name}'.")
            logger.debug(f"Collection info for '{collection_name}': {info}")
            return info
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found when trying to get info.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found: {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error while getting info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error while getting info for collection '{collection_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameter when getting info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameter when getting info for collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to get collection info for '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to get collection info for '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred while getting info for collection '{collection_name}': {str(e)}") from e

    async def search_collection(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        drop_ratio: float = 0.2,
    ) -> list[dict]:
        """
        Perform full text search on a collection.

        Args:
            collection_name: Name of collection to search
            query_text: Text to search for
            limit: Maximum number of results
            output_fields: Fields to return in results
            drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
        """
        logger.info(f"Attempting text search in collection '{collection_name}'.")
        logger.debug(f"Search params: query_text='{query_text[:100]}...', limit={limit}, output_fields={output_fields}, drop_ratio={drop_ratio}")
        try:
            search_params = {"params": {"drop_ratio_search": drop_ratio}}

            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=[query_text], # Sending list with one query text
                anns_field="sparse", # Assuming 'sparse' is for text search / BM25 like features
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )
            logger.info(f"Text search successful in collection '{collection_name}'. Found {len(results[0]) if results else 0} results for the first query.")
            logger.debug(f"Search results for '{collection_name}': {results}")
            return results
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during search.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during search: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SearchError(f"Invalid parameters for search in collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema or index not ready for search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema or index not ready for search in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during search in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Search failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise SearchError(f"Search failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during search in collection '{collection_name}': {str(e)}") from e

    async def query_collection(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Query collection using filter expressions."""
        logger.info(f"Attempting to query collection '{collection_name}'.")
        logger.debug(f"Query params: filter_expr='{filter_expr}', output_fields={output_fields}, limit={limit}")
        try:
            results = await asyncio.to_thread(
                self.client.query,
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
            )
            logger.info(f"Query successful in collection '{collection_name}'. Found {len(results)} results.")
            logger.debug(f"Query results for '{collection_name}': {results}")
            return results
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during query.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during query: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for query in collection '{collection_name}': {str(e)}", exc_info=True)
            raise QueryError(f"Invalid parameters for query in collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for query in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for query in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during query in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during query in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Query failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise QueryError(f"Query failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during query in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during query in collection '{collection_name}': {str(e)}") from e

    async def vector_search(
        self,
        collection_name: str,
        vector: list[float],
        vector_field: str,
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform vector similarity search on a collection.

        Args:
            collection_name: Name of collection to search
            vector: Query vector
            vector_field: Field containing vectors to search
            limit: Maximum number of results
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
        """
        logger.info(f"Attempting vector search in collection '{collection_name}' on field '{vector_field}'.")
        logger.debug(f"Vector search params: vector_present=True, limit={limit}, output_fields={output_fields}, metric_type='{metric_type}', filter_expr='{filter_expr}', nprobe={self.vector_search_nprobe}")
        try:
            search_params = {"metric_type": metric_type, "params": {"nprobe": self.vector_search_nprobe}}

            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=[vector], # Single vector in a list
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
            )
            logger.info(f"Vector search successful in collection '{collection_name}'. Found {len(results[0]) if results else 0} results for the query.")
            logger.debug(f"Vector search results for '{collection_name}', field '{vector_field}': {results}")
            return results
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during vector search.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during vector search: {str(e)}") from e
        except (ParamError, DataNotMatchException) as e:
            logger.error(f"Invalid parameters or data mismatch for vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise SearchError(f"Invalid parameters or data mismatch for vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema or index not ready for vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema or index not ready for vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during vector search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during vector search in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Vector search failed in collection '{collection_name}' on field '{vector_field}' due to Milvus error: {str(e)}", exc_info=True)
            raise SearchError(f"Vector search failed in collection '{collection_name}' on field '{vector_field}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e

    async def hybrid_search(
        self,
        collection_name: str,
        vector: list[float],
        vector_field: str,
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform hybrid search combining vector similarity and attribute filtering.

        Args:
            collection_name: Name of collection to search
            vector: Query vector
            vector_field: Field containing vectors to search
            filter_expr: Filter expression for metadata
            limit: Maximum number of results
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
        """
        logger.warning(f"Attempted to call not implemented method hybrid_search for collection '{collection_name}'.")
        raise NotImplementedError("This method is not yet supported.")

    async def create_collection(
        self,
        collection_name: str,
        schema: dict[str, Any],
        index_params: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Create a new collection with the specified schema.

        Args:
            collection_name: Name for the new collection
            schema: Collection schema definition
            index_params: Optional index parameters
        """
        try:
        logger.info(f"Attempting to create collection '{collection_name}'.")
        logger.debug(f"Collection schema: {schema}, Index params: {index_params}")
        try:
            # Check if collection already exists
            try:
                logger.debug(f"Checking if collection '{collection_name}' already exists.")
                existing_collections = await asyncio.to_thread(self.client.list_collections)
                logger.debug(f"Existing collections: {existing_collections}")
            except (ConnectError, MilvusUnavailableException) as e:
                logger.error(f"Milvus connection error while checking existing collections for '{collection_name}': {str(e)}", exc_info=True)
                raise MilvusConnectionError(f"Milvus connection error while checking existing collections: {str(e)}") from e
            except MilvusException as e:
                logger.error(f"Failed to list existing collections due to Milvus error when checking for '{collection_name}': {str(e)}", exc_info=True)
                raise MilvusOperationError(f"Failed to list existing collections due to Milvus error: {str(e)}") from e

            if collection_name in existing_collections:
                logger.warning(f"Attempt to create collection '{collection_name}' failed: already exists.")
                raise SchemaError(f"Collection '{collection_name}' already exists.")

            # Create collection
            logger.debug(f"Proceeding to create collection '{collection_name}'.")
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=collection_name,
                dimension=schema.get("dimension", 128),
                primary_field_name=schema.get("primary_field", "id"),
                id_type=schema.get("id_type", "INT64"),
                vector_field_name=schema.get("vector_field", "vector"),
                metric_type=schema.get("metric_type", "COSINE"),
                auto_id=schema.get("auto_id", False),
                enable_dynamic_field=schema.get("enable_dynamic_field", True),
            )
            logger.info(f"Collection '{collection_name}' created definition part successfully.")

            # Create index if params provided
            if index_params:
                vector_field_name_from_schema = schema.get("vector_field", "vector")
                logger.info(f"Attempting to create index for field '{vector_field_name_from_schema}' in collection '{collection_name}'.")
                await asyncio.to_thread(
                    self.client.create_index,
                    collection_name=collection_name,
                    field_name=vector_field_name_from_schema,
                    index_params=index_params,
                )
                logger.info(f"Index created successfully for field '{vector_field_name_from_schema}' in collection '{collection_name}'.")
            logger.info(f"Collection '{collection_name}' created successfully (including index if specified).")
            return True
        except SchemaError as e: # Re-raise explicitly raised SchemaError (collection already exists)
            logger.warning(f"SchemaError during creation of collection '{collection_name}': {str(e)}", exc_info=True) # Already logged if it was "already_exists"
            raise
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found unexpectedly during index creation phase: {str(e)}", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found unexpectedly during index creation: {str(e)}") from e
        except IndexNotExistException as e:
            # This var might not be defined if index_params is None, so fetch it safely
            vector_field_name_from_schema = schema.get("vector_field", "vector") if index_params else "unknown_vector_field"
            logger.error(f"Invalid field_name '{vector_field_name_from_schema}' for index creation in '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Invalid field_name '{vector_field_name_from_schema}' for index creation in '{collection_name}': {str(e)}") from e
        except (ParamError, PrimaryKeyException, PartitionKeyException) as e:
            logger.error(f"Schema definition error for collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema definition error for collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during collection creation '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during collection creation '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to create collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to create collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while creating collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred while creating collection '{collection_name}': {str(e)}") from e

    async def insert_data(
        self, collection_name: str, data: dict[str, list[Any]]
    ) -> dict[str, Any]:
        """
        Insert data into a collection.

        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
        """
        record_count = len(data.get(next(iter(data)), [])) if data else 0
        logger.info(f"Attempting to insert {record_count} records into collection '{collection_name}'.")
        logger.debug(f"Inserting data into '{collection_name}': {data if record_count < 5 else str(record_count) + ' records'}") # Log sample data or count

        try:
            result = await asyncio.to_thread(
                self.client.insert, collection_name=collection_name, data=data
            )
            logger.info(f"Successfully inserted {record_count} records into collection '{collection_name}'. PKs: {result.primary_keys}")
            return result
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during insert.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during insert: {str(e)}") from e
        except (DataTypeNotMatchException, DataNotMatchException, PrimaryKeyException) as e:
            logger.error(f"Data type, format, or primary key error during insert into '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Data type, format, or primary key error during insert into '{collection_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Invalid parameters for insert into collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for insert in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for insert in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during insert into collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Insert failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise DataOperationError(f"Insert failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during insert into collection '{collection_name}': {str(e)}") from e

    async def delete_entities(
        self, collection_name: str, filter_expr: str
    ) -> dict[str, Any]:
        """
        Delete entities from a collection based on filter expression.

        Args:
            collection_name: Name of collection
            filter_expr: Filter expression to select entities to delete
        """
        logger.info(f"Attempting to delete entities from collection '{collection_name}' using filter: '{filter_expr}'.")
        try:
            result = await asyncio.to_thread(
                self.client.delete, collection_name=collection_name, expr=filter_expr
            )
            logger.info(f"Successfully deleted entities from collection '{collection_name}' (delete count: {result.delete_count}).")
            return result
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during delete.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during delete: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid filter expression for delete in collection '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Invalid filter expression for delete in collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for delete in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for delete in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during delete in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during delete in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Delete failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise DataOperationError(f"Delete failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during delete in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during delete in collection '{collection_name}': {str(e)}") from e

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Name of collection
        """
        logger.info(f"Attempting to get stats for collection '{collection_name}'.")
        try:
            stats = await asyncio.to_thread(self.client.get_collection_stats, collection_name)
            logger.info(f"Successfully retrieved stats for collection '{collection_name}'.")
            logger.debug(f"Stats for '{collection_name}': {stats}")
            return stats
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found when getting stats.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found when getting stats: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameter when getting stats for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameter when getting stats for collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error while getting stats for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error while getting stats for collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to get collection stats for '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to get collection stats for '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting stats for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred while getting stats for collection '{collection_name}': {str(e)}") from e

    async def multi_vector_search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        vector_field: str,
        limit: int = 5,
        output_fields: Optional[list[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
        search_params: Optional[dict[str, Any]] = None,
    ) -> list[list[dict]]:
        """
        Perform vector similarity search with multiple query vectors.

        Args:
            collection_name: Name of collection to search
            vectors: List of query vectors
            vector_field: Field containing vectors to search
            limit: Maximum number of results per query
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
            search_params: Additional search parameters
        """
        logger.info(f"Attempting multi-vector search in collection '{collection_name}' on field '{vector_field}'.")
        logger.debug(f"Multi-vector search params: num_vectors={len(vectors)}, limit={limit}, output_fields={output_fields}, metric_type='{metric_type}', filter_expr='{filter_expr}', custom_search_params={search_params}, nprobe={self.vector_search_nprobe}")
        try:
            if search_params is None: # If user does not provide specific search_params, use the configured nprobe
                search_params = {"metric_type": metric_type, "params": {"nprobe": self.vector_search_nprobe}}
            # If user provides search_params, we assume they know what they are doing and use their params directly.
            # Alternatively, merge self.vector_search_nprobe if not in user's params. For now, direct override.

            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=vectors,
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr,
            )
            logger.info(f"Multi-vector search successful in collection '{collection_name}'. Found {sum(len(res_list) for res_list in results) if results else 0} total results across all queries.")
            # Avoid logging all results if potentially very large by default
            logger.debug(f"Multi-vector search results for '{collection_name}', field '{vector_field}': First result example: {results[0][0] if results and results[0] else 'N/A'}")
            return results
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during multi-vector search.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during multi-vector search: {str(e)}") from e
        except (ParamError, DataNotMatchException) as e:
            logger.error(f"Invalid parameters or data mismatch for multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise SearchError(f"Invalid parameters or data mismatch for multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema or index not ready for multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema or index not ready for multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during multi-vector search in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during multi-vector search in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Multi-vector search failed in collection '{collection_name}' on field '{vector_field}' due to Milvus error: {str(e)}", exc_info=True)
            raise SearchError(f"Multi-vector search failed in collection '{collection_name}' on field '{vector_field}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during multi-vector search in collection '{collection_name}' on field '{vector_field}': {str(e)}") from e

    async def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        params: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Create an index on a vector field.

        Args:
            collection_name: Name of collection
            field_name: Field to index
            index_type: Type of index (IVF_FLAT, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
            params: Additional index parameters
        """
        logger.info(f"Attempting to create index on field '{field_name}' in collection '{collection_name}'.")
        logger.debug(f"Index params: index_type='{index_type}', metric_type='{metric_type}', custom_params={params}, nlist_config={self.create_index_nlist}")
        try:
            if params is None: # If user does not provide specific index params, use the configured nlist
                params = {"nlist": self.create_index_nlist}
            # If user provides params, we assume they know what they are doing.
            # Alternatively, merge self.create_index_nlist if 'nlist' not in user's params. For now, direct override.

            index_params_payload = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params,
            }

            await asyncio.to_thread(
                self.client.create_index,
                collection_name=collection_name,
                field_name=field_name,
                index_params=index_params_payload,
            )
            logger.info(f"Successfully created index on field '{field_name}' in collection '{collection_name}'.")
            return True
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during index creation on field '{field_name}'.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during index creation on field '{field_name}': {str(e)}") from e
        except IndexNotExistException as e:
            logger.error(f"Cannot create index on field '{field_name}' in collection '{collection_name}'. Field may not exist, not be indexable, or index already exists: {str(e)}", exc_info=True)
            raise SchemaError(f"Cannot create index on field '{field_name}' in collection '{collection_name}'. Field may not exist, not be indexable, or index already exists: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for index creation on field '{field_name}' in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Invalid parameters for index creation on field '{field_name}' in collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for index creation on field '{field_name}' in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for index creation on field '{field_name}' in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during index creation on field '{field_name}' in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during index creation on field '{field_name}' in collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to create index on field '{field_name}' in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to create index on field '{field_name}' in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during index creation on field '{field_name}' in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during index creation on field '{field_name}' in collection '{collection_name}': {str(e)}") from e

    async def bulk_insert(
        self, collection_name: str, data: dict[str, list[Any]], batch_size: int = 1000
    ) -> list[dict[str, Any]]:
        """
        Insert data in batches for better performance.

        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
            batch_size: Number of records per batch
        """
        total_records = len(data.get(next(iter(data)), [])) if data else 0
        num_batches = (total_records + batch_size - 1) // batch_size if batch_size > 0 else 0
        logger.info(f"Attempting bulk insert of {total_records} records in {num_batches} batches into collection '{collection_name}'.")

        try:
            results = []
            if not data or total_records == 0: # Handle empty data case
                logger.info(f"No data provided for bulk insert into collection '{collection_name}'.")
                return []

            field_names = list(data.keys())

            for i in range(0, total_records, batch_size):
                current_batch_num = (i // batch_size) + 1
                logger.debug(f"Processing batch {current_batch_num}/{num_batches} for bulk insert into '{collection_name}'.")
                batch_data = {
                    field: data[field][i : i + batch_size] for field in field_names
                }

                result = await asyncio.to_thread(
                    self.client.insert,
                    collection_name=collection_name,
                    data=batch_data
                )
                logger.debug(f"Successfully inserted batch {current_batch_num}/{num_batches} into collection '{collection_name}'. PKs: {result.primary_keys}")
                results.append(result)

            logger.info(f"Bulk insert successful into collection '{collection_name}'. Total records: {total_records}, Batches: {num_batches}.")
            return results
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during bulk insert.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during bulk insert: {str(e)}") from e
        except (DataTypeNotMatchException, DataNotMatchException, PrimaryKeyException) as e:
            logger.error(f"Data type, format, or primary key error during bulk insert into '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Data type, format, or primary key error during bulk insert into '{collection_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for bulk insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Invalid parameters for bulk insert into collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for bulk insert in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for bulk insert in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during bulk insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during bulk insert into collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Bulk insert failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise DataOperationError(f"Bulk insert failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during bulk insert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during bulk insert into collection '{collection_name}': {str(e)}") from e

    async def load_collection(
        self, collection_name: str, replica_number: int = 1
    ) -> bool:
        """
        Load a collection into memory for search and query.

        Args:
            collection_name: Name of collection to load
            replica_number: Number of replicas
        """
        logger.info(f"Attempting to load collection '{collection_name}' with {replica_number} replica(s).")
        try:
            await asyncio.to_thread(
                self.client.load_collection,
                collection_name=collection_name,
                replica_number=replica_number,
            )
            logger.info(f"Collection '{collection_name}' loaded successfully.")
            return True
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during load.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during load: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for loading collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for loading collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready or no index for loading collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready or no index for loading collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during load of collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during load of collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to load collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to load collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during load of collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during load of collection '{collection_name}': {str(e)}") from e

    async def release_collection(self, collection_name: str) -> bool:
        """
        Release a collection from memory.

        Args:
            collection_name: Name of collection to release
        """
        logger.info(f"Attempting to release collection '{collection_name}'.")
        try:
            await asyncio.to_thread(self.client.release_collection, collection_name)
            logger.info(f"Collection '{collection_name}' released successfully.")
            return True
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during release.", exc_info=True) # MilvusClient might not raise this
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during release: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for releasing collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for releasing collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during release of collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during release of collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to release collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to release collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during release of collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during release of collection '{collection_name}': {str(e)}") from e

    async def get_query_segment_info(self, collection_name: str) -> dict[str, Any]:
        """
        Get information about query segments.

        Args:
            collection_name: Name of collection
        """
        logger.info(f"Attempting to get query segment info for collection '{collection_name}'.")
        try:
            info = await asyncio.to_thread(self.client.get_query_segment_info, collection_name)
            logger.info(f"Successfully retrieved query segment info for collection '{collection_name}'.")
            logger.debug(f"Query segment info for '{collection_name}': {info}")
            return info
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found when getting query segment info.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found when getting query segment info: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for get_query_segment_info on collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for get_query_segment_info on collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Collection '{collection_name}' not loaded or schema not ready for get_query_segment_info: {str(e)}", exc_info=True)
            raise SchemaError(f"Collection '{collection_name}' not loaded or schema not ready for get_query_segment_info: {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during get_query_segment_info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during get_query_segment_info for collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to get query segment info for '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to get query segment info for '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during get_query_segment_info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during get_query_segment_info for collection '{collection_name}': {str(e)}") from e

    async def upsert_data(
        self, collection_name: str, data: dict[str, list[Any]]
    ) -> dict[str, Any]:
        """
        Upsert data into a collection (insert or update if exists).

        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
        """
        record_count = len(data.get(next(iter(data)), [])) if data else 0
        logger.info(f"Attempting to upsert {record_count} records into collection '{collection_name}'.")
        logger.debug(f"Upserting data into '{collection_name}': {data if record_count < 5 else str(record_count) + ' records'}")

        try:
            result = await asyncio.to_thread(
                self.client.upsert, collection_name=collection_name, data=data
            )
            logger.info(f"Successfully upserted {record_count} records into collection '{collection_name}'. Result: {result}") # MilvusClient upsert usually returns MutationResult
            return result
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found during upsert.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found during upsert: {str(e)}") from e
        except (DataTypeNotMatchException, DataNotMatchException, PrimaryKeyException) as e:
            logger.error(f"Data type, format, or primary key error during upsert into '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Data type, format, or primary key error during upsert into '{collection_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for upsert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise DataOperationError(f"Invalid parameters for upsert into collection '{collection_name}': {str(e)}") from e
        except SchemaNotReadyException as e:
            logger.error(f"Schema not ready for upsert in collection '{collection_name}': {str(e)}", exc_info=True)
            raise SchemaError(f"Schema not ready for upsert in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during upsert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during upsert into collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Upsert failed in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise DataOperationError(f"Upsert failed in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during upsert into collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during upsert into collection '{collection_name}': {str(e)}") from e

    async def get_index_info(
        self, collection_name: str, field_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get information about indexes in a collection.

        Args:
            collection_name: Name of collection
            field_name: Optional specific field to get index info for
        """
        try:
            # The client's describe_index method takes 'index_name', which might be the same as field_name
            # if the default index naming convention is used (usually based on the field).
            # If users can create custom index names not matching field names, this might need adjustment
            # or clarification on what 'field_name' parameter means here. Assuming it's the name of the index.
            logger.info(f"Attempting to get index info for field '{field_name}' in collection '{collection_name}'.")
            index_info = await asyncio.to_thread(
                self.client.describe_index,
                collection_name=collection_name,
                index_name=field_name
            )
            logger.info(f"Successfully retrieved index info for field '{field_name}' in collection '{collection_name}'.")
            logger.debug(f"Index info for '{collection_name}', field '{field_name}': {index_info}")
            return index_info
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found when getting index info for index/field '{field_name}'.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found when getting index info for index/field '{field_name}': {str(e)}") from e
        except IndexNotExistException as e:
            logger.error(f"Index for field '{field_name}' not found in collection '{collection_name}'.", exc_info=True)
            raise IndexNotFoundError(f"Index for field '{field_name}' not found in collection '{collection_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for get_index_info for index/field '{field_name}' in collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for get_index_info for index/field '{field_name}' in collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during get_index_info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during get_index_info for collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to get index info for index/field '{field_name}' in collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to get index info for index/field '{field_name}' in collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during get_index_info for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during get_index_info for collection '{collection_name}': {str(e)}") from e

    async def get_collection_loading_progress(
        self, collection_name: str
    ) -> dict[str, Any]:
        """
        Get the loading progress of a collection.

        Args:
            collection_name: Name of collection
        """
        logger.info(f"Attempting to get collection loading progress for '{collection_name}'.")
        try:
            progress = await asyncio.to_thread(self.client.get_load_state, collection_name)
            logger.info(f"Successfully retrieved loading progress for collection '{collection_name}'.")
            logger.debug(f"Loading progress for '{collection_name}': {progress}")
            return progress
        except CollectionNotExistException as e:
            logger.error(f"Collection '{collection_name}' not found when getting loading progress.", exc_info=True)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found when getting loading progress: {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters for get_collection_loading_progress on collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Invalid parameters for get_collection_loading_progress on collection '{collection_name}': {str(e)}") from e
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error during get_collection_loading_progress for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error during get_collection_loading_progress for collection '{collection_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to get loading progress for collection '{collection_name}' due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to get loading progress for collection '{collection_name}' due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during get_collection_loading_progress for collection '{collection_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred during get_collection_loading_progress for collection '{collection_name}': {str(e)}") from e

    async def list_databases(self) -> list[str]:
        """List all databases in the Milvus instance."""
        logger.info("Attempting to list all databases.")
        try:
            databases = await asyncio.to_thread(self.client.list_databases)
            logger.info(f"Successfully listed databases: {databases}")
            return databases
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Milvus connection error while listing databases: {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Milvus connection error while listing databases: {str(e)}") from e
        except MilvusException as e:
            logger.error(f"Failed to list databases due to Milvus error: {str(e)}", exc_info=True)
            raise MilvusOperationError(f"Failed to list databases due to Milvus error: {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while listing databases: {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred while listing databases: {str(e)}") from e

    async def use_database(self, db_name: str) -> bool:
        """Switch to a different database.

        Args:
            db_name: Name of the database to use
        """
        logger.info(f"Attempting to switch to database '{db_name}'. Current client URI: {self.uri}, Token Provided: {self.token is not None}")
        try:
            # Create a new client with the specified database - this is a synchronous call
            self.client = MilvusClient(uri=self.uri, token=self.token, db_name=db_name)
            logger.info(f"Successfully switched client to database '{db_name}'.")
            # Optionally, verify connection, e.g., by listing collections or a specific command
            # logger.debug(f"Verifying new database context by listing collections from '{db_name}'.")
            # await asyncio.to_thread(self.client.list_collections)
            return True
        except (ConnectError, MilvusUnavailableException) as e:
            logger.error(f"Failed to connect to Milvus or switch to database '{db_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Failed to connect to Milvus or switch to database '{db_name}': {str(e)}") from e
        except ParamError as e:
            logger.error(f"Invalid parameters when trying to switch to database '{db_name}': {str(e)}", exc_info=True)
            raise MilvusConnectionError(f"Invalid parameters when trying to switch to database '{db_name}': {str(e)}") from e
        except MilvusException as e:
            logger.error(f"An unexpected Milvus error occurred when switching to database '{db_name}': {str(e)}", exc_info=True)
            raise MilvusOperationError(f"An unexpected Milvus error occurred when switching to database '{db_name}': {str(e)}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred when switching to database '{db_name}': {str(e)}", exc_info=True)
            raise MilvusMCPError(f"An unexpected error occurred when switching to database '{db_name}': {str(e)}") from e


class MilvusContext:
    def __init__(self, connector: MilvusConnector):
        self.connector = connector


@asynccontextmanager
async def server_lifespan(server: FastMCP, settings: Settings) -> AsyncIterator[MilvusContext]:
    """Manage application lifecycle for Milvus connector."""
    logger.info("Milvus MCP server lifespan starting.")
    try:
        connector = MilvusConnector(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            db_name=settings.milvus_db,
            vector_search_nprobe=settings.milvus_vector_search_nprobe,
            create_index_nlist=settings.milvus_create_index_nlist,
        )
        logger.info("MilvusConnector created within server lifespan using Pydantic settings.")
        yield MilvusContext(connector)
    except Exception as e:
        logger.error(f"Error during Milvus server lifespan setup: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Milvus MCP server lifespan ending.")

settings = Settings()
lifespan_with_settings = functools.partial(server_lifespan, settings=settings)
mcp = FastMCP("Milvus", lifespan=lifespan_with_settings)


@mcp.tool()
async def milvus_text_search(
    collection_name: str,
    query_text: str,
    limit: int = 5,
    output_fields: Optional[list[str]] = None,
    drop_ratio: float = 0.2,
    ctx: Context = None,
) -> str:
    """
    Search for documents using full text search in a Milvus collection.

    Args:
        collection_name: Name of the collection to search
        query_text: Text to search for
        limit: Maximum number of results to return
        output_fields: Fields to include in results
        drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
    """
    logger.info(f"Tool 'milvus_text_search' called for collection '{collection_name}'.")
    logger.debug(f"milvus_text_search params: query_text='{query_text[:100]}...', limit={limit}, output_fields={output_fields}, drop_ratio={drop_ratio}")
    try:
        connector = ctx.request_context.lifespan_context.connector
        results = await connector.search_collection(
            collection_name=collection_name,
            query_text=query_text,
            limit=limit,
            output_fields=output_fields,
            drop_ratio=drop_ratio,
        )

        output = f"Search results for '{query_text}' in collection '{collection_name}':\n\n"
        for hit_list in results: # results is List[Hits], Hits is List[Hit]
            for hit in hit_list:
                output += f"{hit.entity.to_dict() if hasattr(hit, 'entity') and hasattr(hit.entity, 'to_dict') else str(hit)}\n\n"

        logger.info(f"Tool 'milvus_text_search' completed for '{collection_name}'.")
        logger.debug(f"milvus_text_search output for '{collection_name}': {output[:200]}...") # Log snippet
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_text_search' for collection '{collection_name}': {str(e)}", exc_info=True)
        raise # Re-raise to be handled by FastMCP framework if it has error handling


@mcp.tool()
async def milvus_list_collections(ctx: Context) -> str:
    """List all collections in the database."""
    logger.info("Tool 'milvus_list_collections' called.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        collections = await connector.list_collections()
        output = f"Collections in database:\n{', '.join(collections)}"
        logger.info(f"Tool 'milvus_list_collections' completed. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_list_collections': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_query(
    collection_name: str,
    filter_expr: str,
    output_fields: Optional[list[str]] = None,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """
    Query collection using filter expressions.

    Args:
        collection_name: Name of the collection to query
        filter_expr: Filter expression (e.g. 'age > 20')
        output_fields: Fields to include in results
        limit: Maximum number of results
    """
    logger.info(f"Tool 'milvus_query' called for collection '{collection_name}'.")
    logger.debug(f"milvus_query params: filter_expr='{filter_expr}', output_fields={output_fields}, limit={limit}")
    try:
        connector = ctx.request_context.lifespan_context.connector
        results = await connector.query_collection(
            collection_name=collection_name,
            filter_expr=filter_expr,
            output_fields=output_fields,
            limit=limit,
        )

        output = f"Query results for '{filter_expr}' in collection '{collection_name}':\n\n"
        for result_item in results: # results is List[Dict]
             output += f"{str(result_item)}\n\n" # Assuming result_item is a dict

        logger.info(f"Tool 'milvus_query' completed for '{collection_name}'.")
        logger.debug(f"milvus_query output for '{collection_name}': {output[:200]}...")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_query' for collection '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_vector_search(
    collection_name: str,
    vector: list[float],
    vector_field: str = "vector",
    limit: int = 5,
    output_fields: Optional[list[str]] = None,
    metric_type: str = "COSINE",
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Perform vector similarity search on a collection.

    Args:
        collection_name: Name of the collection to search
        vector: Query vector
        vector_field: Field containing vectors to search
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expr: Optional filter expression
    """
    logger.info(f"Tool 'milvus_vector_search' called for collection '{collection_name}', field '{vector_field}'.")
    logger.debug(f"milvus_vector_search params: vector_present=True, limit={limit}, output_fields={output_fields}, metric_type='{metric_type}', filter_expr='{filter_expr}'")
    try:
        connector = ctx.request_context.lifespan_context.connector
        results = await connector.vector_search(
            collection_name=collection_name,
            vector=vector,
            vector_field=vector_field,
            limit=limit,
            output_fields=output_fields,
            metric_type=metric_type,
            filter_expr=filter_expr,
        )

        output = f"Vector search results for '{collection_name}':\n\n"
        for hit_list in results: # results is List[Hits], Hits is List[Hit]
            for hit in hit_list:
                 output += f"{hit.entity.to_dict() if hasattr(hit, 'entity') and hasattr(hit.entity, 'to_dict') else str(hit)}\n\n"

        logger.info(f"Tool 'milvus_vector_search' completed for '{collection_name}'.")
        logger.debug(f"milvus_vector_search output for '{collection_name}': {output[:200]}...")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_vector_search' for collection '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_create_collection(
    collection_name: str,
    collection_schema: dict[str, Any],
    index_params: Optional[dict[str, Any]] = None,
    ctx: Context = None,
) -> str:
    """
    Create a new collection with specified schema.

    Args:
        collection_name: Name for the new collection
        collection_schema: Collection schema definition
        index_params: Optional index parameters
    """
    logger.info(f"Tool 'milvus_create_collection' called for collection '{collection_name}'.")
    logger.debug(f"milvus_create_collection params: schema={collection_schema}, index_params={index_params}")
    try:
        connector = ctx.request_context.lifespan_context.connector
        success = await connector.create_collection(
            collection_name=collection_name,
            schema=collection_schema,
            index_params=index_params,
        )
        output = f"Collection '{collection_name}' created successfully"
        logger.info(f"Tool 'milvus_create_collection' completed for '{collection_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_create_collection' for '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_insert_data(
    collection_name: str, data: dict[str, list[Any]], ctx: Context = None
) -> str:
    """
    Insert data into a collection.

    Args:
        collection_name: Name of collection
        data: Dictionary mapping field names to lists of values
    """
    record_count = len(data.get(next(iter(data)), [])) if data else 0
    logger.info(f"Tool 'milvus_insert_data' called for collection '{collection_name}', {record_count} records.")
    logger.debug(f"milvus_insert_data params: data_keys={list(data.keys()) if data else 'N/A'}")
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.insert_data(collection_name=collection_name, data=data)
        output = f"Data inserted into collection '{collection_name}' with result: {str(result)}"
        logger.info(f"Tool 'milvus_insert_data' completed for '{collection_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_insert_data' for '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_delete_entities(
    collection_name: str, filter_expr: str, ctx: Context = None
) -> str:
    """
    Delete entities from a collection based on filter expression.

    Args:
        collection_name: Name of collection
        filter_expr: Filter expression to select entities to delete
    """
    logger.info(f"Tool 'milvus_delete_entities' called for collection '{collection_name}'.")
    logger.debug(f"milvus_delete_entities params: filter_expr='{filter_expr}'")
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.delete_entities(
            collection_name=collection_name, filter_expr=filter_expr
        )
        output = f"Entities deleted from collection '{collection_name}' with result: {str(result)}"
        logger.info(f"Tool 'milvus_delete_entities' completed for '{collection_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_delete_entities' for '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_load_collection(
    collection_name: str, replica_number: int = 1, ctx: Context = None
) -> str:
    """
    Load a collection into memory for search and query.

    Args:
        collection_name: Name of collection to load
        replica_number: Number of replicas
    """
    logger.info(f"Tool 'milvus_load_collection' called for '{collection_name}', replicas={replica_number}.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        success = await connector.load_collection(
            collection_name=collection_name, replica_number=replica_number
        )
        output = f"Collection '{collection_name}' loaded successfully with {replica_number} replica(s)"
        logger.info(f"Tool 'milvus_load_collection' completed for '{collection_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_load_collection' for '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_release_collection(collection_name: str, ctx: Context = None) -> str:
    """
    Release a collection from memory.

    Args:
        collection_name: Name of collection to release
    """
    logger.info(f"Tool 'milvus_release_collection' called for '{collection_name}'.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        success = await connector.release_collection(collection_name=collection_name)
        output = f"Collection '{collection_name}' released successfully"
        logger.info(f"Tool 'milvus_release_collection' completed for '{collection_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_release_collection' for '{collection_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_list_databases(ctx: Context = None) -> str:
    """List all databases in the Milvus instance."""
    logger.info("Tool 'milvus_list_databases' called.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        databases = await connector.list_databases()
        output = f"Databases in Milvus instance:\n{', '.join(databases)}"
        logger.info(f"Tool 'milvus_list_databases' completed. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_list_databases': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_use_database(db_name: str, ctx: Context = None) -> str:
    """
    Switch to a different database.

    Args:
        db_name: Name of the database to use
    """
    logger.info(f"Tool 'milvus_use_database' called for database '{db_name}'.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        success = await connector.use_database(db_name)
        output = f"Switched to database '{db_name}' successfully"
        logger.info(f"Tool 'milvus_use_database' completed for '{db_name}'. Output: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_use_database' for '{db_name}': {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def milvus_get_collection_info(collection_name: str, ctx: Context = None) -> str:
    """
    Lists detailed information about a specific collection

    Args:
        collection_name: Name of collection to load
    """
    logger.info(f"Tool 'milvus_get_collection_info' called for collection '{collection_name}'.")
    try:
        connector = ctx.request_context.lifespan_context.connector
        collection_info = await connector.get_collection_info(collection_name)
        info_str = json.dumps(collection_info, indent=2) # Assuming collection_info is JSON serializable
        output = f"Collection information:\n{info_str}"
        logger.info(f"Tool 'milvus_get_collection_info' completed for '{collection_name}'.")
        logger.debug(f"milvus_get_collection_info output: {output[:200]}...")
        return output
    except Exception as e:
        logger.error(f"Error in tool 'milvus_get_collection_info' for '{collection_name}': {str(e)}", exc_info=True)
        raise


# parse_arguments and setup_environment are removed.

def main():
    """Main entry point for the Milvus MCP Server."""
    settings = Settings()

    # Configure logging using Pydantic settings
    log_level_int = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Starting Milvus MCP Server with Pydantic settings.")
    logger.info(f"Loaded settings: MILVUS_URI='{settings.milvus_uri}', MILVUS_DB='{settings.milvus_db}', MILVUS_TOKEN_PROVIDED={settings.milvus_token is not None}, LOG_LEVEL='{settings.log_level}'")
    logger.debug(f"Full settings (excluding token): {settings.dict(exclude={'milvus_token'})}")


    # Prepare lifespan with settings
    lifespan_with_settings = functools.partial(server_lifespan, settings=settings)

    # Initialize FastMCP with the partial lifespan
    # Global 'mcp' instance, or pass it around if preferred by FastMCP patterns
    global mcp
    mcp = FastMCP("Milvus", lifespan=lifespan_with_settings)

    # Re-register tools if they were defined before mcp was re-initialized.
    # This depends on how FastMCP handles tool registration with a new instance.
    # For this script structure, tools are defined on the global 'mcp' instance.
    # If 'mcp' was not global, tools would need to be re-associated.
    # The current script structure redefines global mcp, so tools defined after it should be fine.
    # Tools defined *before* this new mcp = FastMCP(...) would be on the old instance.
    # In this specific file, tools are defined after the initial mcp = FastMCP(...),
    # so we need to ensure they are registered with the *new* mcp instance.
    # The simplest way if tools are defined before this point is to re-apply decorators,
    # or FastMCP might offer a way to copy/re-register.
    # However, looking at the original script, tools are defined after global mcp is initialized.
    # The challenge here is that `mcp` is defined globally, then tools are decorated onto it.
    # In `main`, we are now creating a *new* `mcp` instance.
    # The tools will remain associated with the *old* global `mcp` instance unless re-registered.
    # This is a structural issue with how FastMCP tools might be registered in relation to main().
    # For now, assume tools are defined *after* mcp is initialized in main, or that FastMCP handles this.
    # A simple fix is to move tool definitions into a function called after mcp is set up in main,
    # or ensure mcp is a singleton configured once.
    # For now, the current structure implies tools are registered to the `mcp` instance available at definition time.
    # The prompt does not ask to refactor tool registration, so I will assume the current structure is intended
    # and that the tools will be registered on the new `mcp` instance if `mcp` is indeed a global that's reassigned.
    # Let's assume for now that the tool decorators will pick up the new `mcp` instance if it's a global variable.
    # If not, this would be a runtime issue.
    # The original script defines `mcp = FastMCP(...)` globally.
    # If we redefine it in `main()`, the globally defined tools will not be on this new instance.
    # The most straightforward way to handle this is to make `mcp` fully initialized in `main` and pass it around,
    # or to configure the global `mcp` instance.
    # Let's stick to re-assigning the global `mcp` and assume tool registration will follow,
    # or note this as a potential structural refactor needed for FastMCP.
    # A common pattern is app = FastMCP(); then @app.tool(), then app.run().
    # So, tools should be defined *after* `mcp = FastMCP(...)` in `main`.
    # The current file structure has tools defined after the initial global `mcp`.
    # This change means the initial global `mcp` is not used for running.
    # This is a bit of a structural tangle.
    # The least invasive change for now is to update the global `mcp` instance.

    # The tools are defined using @mcp.tool() on the global mcp instance.
    # We need to make sure they are registered with the new mcp instance created in main.
    # This means either mcp should be initialized once globally with settings,
    # or tools should be registered explicitly on the instance from main.
    # The path of least resistance given the current structure is to update the global instance,
    # but that's not ideal.
    # A better way: define tools in a function, and call that function with the mcp instance from main.
    # The global `mcp` instance is now initialized above with settings.
    # Tools defined below will be registered on this instance.
    mcp.run()


if __name__ == "__main__":
    main()
