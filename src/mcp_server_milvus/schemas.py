"""Request and response schemas for MCP tools.

This module defines Pydantic models for requests and responses corresponding
to each tool exposed by the Milvus MCP server. These models can be used
for validation, serialization and clearer typing across the codebase.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MilvusTextSearchRequest(BaseModel):
    """Request model for text search."""

    collection_name: str = Field(..., description="Name of the collection to search", min_length=1)
    query_text: str = Field(..., description="Text query for BM25 search", min_length=1)
    limit: int = Field(5, ge=1, le=1000, description="Maximum number of results to return")
    output_fields: Optional[list[str]] = Field(None, description="Fields to include in results")
    drop_ratio: float = Field(0.2, ge=0.0, le=1.0, description="Proportion of low-frequency terms to ignore")


class MilvusTextSearchResponse(BaseModel):
    """Response model for text search."""

    results: list[str] = Field(..., description="Search results serialized to strings")


class MilvusListCollectionsRequest(BaseModel):
    """Request model for listing collections."""


class MilvusListCollectionsResponse(BaseModel):
    """Response model for listing collections."""

    collections: list[str] = Field(..., description="List of collection names")


class MilvusQueryRequest(BaseModel):
    """Request model for querying collections."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    filter_expr: str = Field(..., description="Filter expression", min_length=1)
    output_fields: Optional[list[str]] = Field(None, description="Fields to include in results")
    limit: int = Field(10, ge=1, le=10000, description="Maximum number of results")


class MilvusQueryResponse(BaseModel):
    """Response model for query operations."""

    results: list[str] = Field(..., description="Query results")


class MilvusVectorSearchRequest(BaseModel):
    """Request model for vector search."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    vector: list[float] = Field(..., description="Query vector", min_length=1)
    vector_field: str = Field("vector", description="Field containing vectors", min_length=1)
    limit: int = Field(5, ge=1, le=1000, description="Maximum number of results")
    output_fields: Optional[list[str]] = Field(None, description="Fields to include in results")
    metric_type: str = Field("COSINE", description="Distance metric (COSINE, L2, IP)")
    filter_expr: Optional[str] = Field(None, description="Optional filter expression")


class MilvusVectorSearchResponse(BaseModel):
    """Response model for vector search."""

    results: list[str] = Field(..., description="Search results")


class MilvusHybridSearchRequest(BaseModel):
    """Request model for hybrid search."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    query_text: str = Field(..., description="Text query for BM25 search", min_length=1)
    text_field: str = Field(..., description="Field name for text search", min_length=1)
    vector: list[float] = Field(..., description="Query vector", min_length=1)
    vector_field: str = Field(..., description="Field name for vector search", min_length=1)
    limit: int = Field(5, ge=1, le=1000, description="Maximum number of results")
    output_fields: Optional[list[str]] = Field(None, description="Fields to include in results")
    filter_expr: Optional[str] = Field(None, description="Optional filter expression")


class MilvusHybridSearchResponse(BaseModel):
    """Response model for hybrid search."""

    results: list[str] = Field(..., description="Search results")


class MilvusCreateCollectionRequest(BaseModel):
    """Request model for creating collections."""

    collection_name: str = Field(..., description="Name for the new collection", min_length=1)
    collection_schema: dict[str, Any] = Field(..., description="Collection schema definition")
    index_params: Optional[dict[str, Any]] = Field(None, description="Optional index parameters")


class MilvusCreateCollectionResponse(BaseModel):
    """Response model for collection creation."""

    message: str = Field(..., description="Status message")


class MilvusInsertDataRequest(BaseModel):
    """Request model for inserting data."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    data: list[dict[str, Any]] = Field(..., description="List of records to insert", min_length=1)


class MilvusInsertDataResponse(BaseModel):
    """Response model for data insertion."""

    message: str = Field(..., description="Status message")


class MilvusDeleteEntitiesRequest(BaseModel):
    """Request model for deleting entities."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    filter_expr: str = Field(..., description="Filter expression", min_length=1)


class MilvusDeleteEntitiesResponse(BaseModel):
    """Response model for entity deletion."""

    message: str = Field(..., description="Status message")


class MilvusLoadCollectionRequest(BaseModel):
    """Request model for loading collections."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)
    replica_number: int = Field(1, ge=1, description="Number of replicas")


class MilvusLoadCollectionResponse(BaseModel):
    """Response model for collection loading."""

    message: str = Field(..., description="Status message")


class MilvusReleaseCollectionRequest(BaseModel):
    """Request model for releasing collections."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)


class MilvusReleaseCollectionResponse(BaseModel):
    """Response model for collection releasing."""

    message: str = Field(..., description="Status message")


class MilvusListDatabasesRequest(BaseModel):
    """Request model for listing databases."""


class MilvusListDatabasesResponse(BaseModel):
    """Response model for listing databases."""

    databases: list[str] = Field(..., description="List of database names")


class MilvusUseDatabaseRequest(BaseModel):
    """Request model for switching databases."""

    db_name: str = Field(..., description="Name of the database", min_length=1)


class MilvusUseDatabaseResponse(BaseModel):
    """Response model for database switching."""

    message: str = Field(..., description="Status message")


class MilvusGetCollectionInfoRequest(BaseModel):
    """Request model for getting collection information."""

    collection_name: str = Field(..., description="Name of the collection", min_length=1)


class MilvusGetCollectionInfoResponse(BaseModel):
    """Response model for collection information."""

    info: dict[str, Any] = Field(..., description="Collection information")


__all__ = [
    "MilvusTextSearchRequest",
    "MilvusTextSearchResponse",
    "MilvusListCollectionsRequest",
    "MilvusListCollectionsResponse",
    "MilvusQueryRequest",
    "MilvusQueryResponse",
    "MilvusVectorSearchRequest",
    "MilvusVectorSearchResponse",
    "MilvusHybridSearchRequest",
    "MilvusHybridSearchResponse",
    "MilvusCreateCollectionRequest",
    "MilvusCreateCollectionResponse",
    "MilvusInsertDataRequest",
    "MilvusInsertDataResponse",
    "MilvusDeleteEntitiesRequest",
    "MilvusDeleteEntitiesResponse",
    "MilvusLoadCollectionRequest",
    "MilvusLoadCollectionResponse",
    "MilvusReleaseCollectionRequest",
    "MilvusReleaseCollectionResponse",
    "MilvusListDatabasesRequest",
    "MilvusListDatabasesResponse",
    "MilvusUseDatabaseRequest",
    "MilvusUseDatabaseResponse",
    "MilvusGetCollectionInfoRequest",
    "MilvusGetCollectionInfoResponse",
]
