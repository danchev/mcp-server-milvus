"""Basic tests for the MCP tool request/response schemas.

Sanity checks to verify models can be instantiated and serialized.
"""

from mcp_server_milvus import schemas


def test_text_search_schema():
    req = schemas.MilvusTextSearchRequest(collection_name="col", query_text="hello")
    assert req.limit == 5
    resp = schemas.MilvusTextSearchResponse(results=["r1", "r2"])
    assert resp.results == ["r1", "r2"]


def test_list_collections_schema():
    resp = schemas.MilvusListCollectionsResponse(collections=["a", "b"])
    assert resp.collections == ["a", "b"]


def test_query_schema():
    req = schemas.MilvusQueryRequest(collection_name="c", filter_expr="x > 10")
    assert req.limit == 10
    resp = schemas.MilvusQueryResponse(results=["row1"])
    assert isinstance(resp.results, list)


def test_vector_search_schema():
    req = schemas.MilvusVectorSearchRequest(collection_name="c", vector=[0.1, 0.2])
    assert req.vector == [0.1, 0.2]
    resp = schemas.MilvusVectorSearchResponse(results=["hit1"])
    assert resp.results[0] == "hit1"


def test_hybrid_search_schema():
    req = schemas.MilvusHybridSearchRequest(
        collection_name="c",
        query_text="q",
        text_field="sparse",
        vector=[0.1],
        vector_field="vector",
    )
    assert req.query_text == "q"


def test_create_insert_delete_load_release_schemas():
    create_req = schemas.MilvusCreateCollectionRequest(collection_name="c", collection_schema={"dimension": 128})
    assert create_req.collection_name == "c"
    ins_req = schemas.MilvusInsertDataRequest(collection_name="c", data=[{"id": 1}])
    assert isinstance(ins_req.data, list)
    del_req = schemas.MilvusDeleteEntitiesRequest(collection_name="c", filter_expr="id == 1")
    assert del_req.filter_expr == "id == 1"
    load_req = schemas.MilvusLoadCollectionRequest(collection_name="c")
    assert load_req.replica_number == 1
    rel_req = schemas.MilvusReleaseCollectionRequest(collection_name="c")
    assert rel_req.collection_name == "c"


def test_database_schemas():
    resp = schemas.MilvusListDatabasesResponse(databases=["db1"])  # simple typing
    assert resp.databases == ["db1"]
    use_req = schemas.MilvusUseDatabaseRequest(db_name="db1")
    assert use_req.db_name == "db1"


def test_get_collection_info_schema():
    req = schemas.MilvusGetCollectionInfoRequest(collection_name="c")
    assert req.collection_name == "c"
    resp = schemas.MilvusGetCollectionInfoResponse(info={"name": "c"})
    assert resp.info["name"] == "c"
