"""Tests for server module covering MilvusConnector and MCP tool wrappers.

These tests monkeypatch external dependencies so they don't perform network I/O.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mcp_server_milvus import server


class FakeClient:
    def __init__(self, uri=None, token=None, db_name=None):
        self.uri = uri
        self.token = token
        self.db_name = db_name

    def list_collections(self):
        return ["c1", "c2"]

    def describe_collection(self, name):
        return {"name": name}

    def search(self, **kwargs):
        return ["hit1"]

    def query(self, **kwargs):
        return [{"id": 1}]

    def hybrid_search(self, **kwargs):
        return ["hybrid1"]

    def create_collection(self, **kwargs):
        return True

    def create_index(self, **kwargs):
        return True

    def insert(self, **kwargs):
        return {"inserted": True}

    def delete(self, **kwargs):
        return {"deleted": True}

    def get_collection_stats(self, *a, **kw):
        return {"stats": {}}

    def upsert(self, **kwargs):
        return {"upserted": True}

    def describe_index(self, **kwargs):
        return {"index": {}}

    def get_load_state(self, *a, **kw):
        return {"progress": 100}

    def list_databases(self):
        return ["db1"]

    def load_collection(self, **kwargs):
        return None

    def release_collection(self, **kwargs):
        return None


class FakeIndexParams:
    @classmethod
    def add_index(cls, **kwargs):
        return {"idx": True}


class FakeAnnSearchRequest:
    def __init__(self, **_):
        pass


class FakeRRFRanker:
    def __init__(self, *_, **__):
        pass


def make_ctx_with_connector(connector) -> SimpleNamespace:
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=SimpleNamespace(connector=connector)))


@pytest.fixture
def mock_milvus_deps(monkeypatch):
    """Fixture to mock Milvus dependencies."""
    monkeypatch.setattr(server, "MilvusClient", FakeClient)
    monkeypatch.setattr(server, "IndexParams", FakeIndexParams)
    monkeypatch.setattr(server, "AnnSearchRequest", FakeAnnSearchRequest)
    monkeypatch.setattr(server, "RRFRanker", FakeRRFRanker)
    monkeypatch.setattr(server, "utility", SimpleNamespace(get_query_segment_info=lambda name: {"segments": []}))


@pytest.fixture
def milvus_connector(mock_milvus_deps):
    """Fixture to create a MilvusConnector with mocked dependencies."""
    return server.MilvusConnector(uri="u://x", token="t")


@pytest.fixture
def mcp_context(milvus_connector):
    """Fixture to create an MCP context with a connector."""
    return make_ctx_with_connector(milvus_connector)


def test_milvus_connector_from_config(mock_milvus_deps):
    """Test MilvusConnector.from_config classmethod."""
    from mcp_server_milvus.config import Settings

    # Test with explicit settings
    settings = Settings(milvus_uri="http://test:9000", milvus_token="token", milvus_db="testdb")
    connector = server.MilvusConnector.from_config(settings)
    assert connector.uri == "http://test:9000"
    assert connector.token == "token"
    assert connector.db_name == "testdb"

    # Test with None (uses get_settings)
    connector2 = server.MilvusConnector.from_config(None)
    assert connector2.uri is not None


def test_milvus_connector_basic_methods(milvus_connector):
    res = asyncio.run(milvus_connector.list_collections())
    assert res == ["c1", "c2"]
    info = asyncio.run(milvus_connector.get_collection_info("c1"))
    assert info["name"] == "c1"
    search_res = asyncio.run(milvus_connector.search_collection("c1", "query"))
    assert search_res == ["hit1"]
    q = asyncio.run(milvus_connector.query_collection("c1", "id==1"))
    assert q[0]["id"] == 1
    v = asyncio.run(milvus_connector.vector_search("c1", [0.1, 0.2], "vec"))
    assert v == ["hit1"]
    h = asyncio.run(milvus_connector.hybrid_search("c1", "q", "text", [0.1], "vec", limit=1))
    assert h == ["hybrid1"]
    created = asyncio.run(milvus_connector.create_collection("c3", {"dimension": 128}))
    assert created is True
    inserted = asyncio.run(milvus_connector.insert_data("c1", [{"id": 1}]))
    assert inserted["inserted"] is True
    deleted = asyncio.run(milvus_connector.delete_entities("c1", "id==1"))
    assert deleted["deleted"] is True
    stats = asyncio.run(milvus_connector.get_collection_stats("c1"))
    assert "stats" in stats
    m = asyncio.run(milvus_connector.multi_vector_search("c1", [[0.1, 0.2]], "vec"))
    assert m == ["hit1"]
    index_ok = asyncio.run(milvus_connector.create_index("c1", "vec"))
    assert index_ok is True
    bulk = asyncio.run(milvus_connector.bulk_insert("c1", {"id": [1, 2]}))
    assert isinstance(bulk, list)
    l_ = asyncio.run(milvus_connector.load_collection("c1"))
    assert l_ is True
    r = asyncio.run(milvus_connector.release_collection("c1"))
    assert r is True
    seg = asyncio.run(milvus_connector.get_query_segment_info("c1"))
    assert seg == {"segments": []} or isinstance(seg, dict)
    upserted = asyncio.run(milvus_connector.upsert_data("c1", {"id": [1]}))
    assert upserted["upserted"] is True
    idx_info = asyncio.run(milvus_connector.get_index_info("c1"))
    assert "index" in idx_info
    progress = asyncio.run(milvus_connector.get_collection_loading_progress("c1"))
    assert progress["progress"] == 100
    dbs = asyncio.run(milvus_connector.list_databases())
    assert dbs == ["db1"]
    assert asyncio.run(milvus_connector.use_database("db1")) is True


def test_mcp_tool_wrappers(mcp_context):
    # Call a subset of MCP tools (async functions) via asyncio
    collections_resp = asyncio.run(server.milvus_list_collections(mcp_context))
    assert collections_resp.collections == ["c1", "c2"]
    text_search_resp = asyncio.run(server.milvus_text_search(mcp_context, "c1", "q"))
    assert text_search_resp.results == ["hit1"]
    query_resp = asyncio.run(server.milvus_query(mcp_context, "c1", "id==1"))
    assert query_resp.results == ["{'id': 1}"] or isinstance(query_resp.results, list)
    vector_resp = asyncio.run(server.milvus_vector_search(mcp_context, "c1", [0.1]))
    assert vector_resp.results == ["hit1"]
    hybrid_resp = asyncio.run(server.milvus_hybrid_search(mcp_context, "c1", "q", "text", [0.1], "vec"))
    assert hybrid_resp.results == ["hybrid1"]
    # Create collection success path
    create_resp = asyncio.run(server.milvus_create_collection(mcp_context, "c3", {"dimension": 1}))
    assert "created successfully" in create_resp.message
    # Insert/delete/load/release
    insert_resp = asyncio.run(server.milvus_insert_data(mcp_context, "c1", [{"id": 1}]))
    assert "Data inserted" in insert_resp.message
    delete_resp = asyncio.run(server.milvus_delete_entities(mcp_context, "c1", "id==1"))
    assert "Entities deleted" in delete_resp.message
    load_resp = asyncio.run(server.milvus_load_collection(mcp_context, "c1"))
    assert "loaded successfully" in load_resp.message
    release_resp = asyncio.run(server.milvus_release_collection(mcp_context, "c1"))
    assert "released successfully" in release_resp.message
    dbs_resp = asyncio.run(server.milvus_list_databases(mcp_context))
    assert dbs_resp.databases == ["db1"]
    use_db_resp = asyncio.run(server.milvus_use_database(mcp_context, "db1"))
    assert "Switched to database" in use_db_resp.message
    info_resp = asyncio.run(server.milvus_get_collection_info(mcp_context, "c1"))
    assert "name" in info_resp.info


def test_mcp_tool_create_collection_with_index(mcp_context):
    """Test create_collection with index_params."""
    # Create collection with index
    create_resp = asyncio.run(
        server.milvus_create_collection(
            mcp_context,
            "c4",
            {"dimension": 128, "vector_field": "vec"},
            {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
    )
    assert "created successfully" in create_resp.message


def test_milvus_connector_validate_collection_exists(mock_milvus_deps, monkeypatch):
    """Test that creating an existing collection raises error."""

    class FakeClientWithC1(FakeClient):
        def list_collections(self):
            return ["c1"]

    monkeypatch.setattr(server, "MilvusClient", FakeClientWithC1)
    monkeypatch.setattr(server, "IndexParams", FakeIndexParams)
    connector = server.MilvusConnector(uri="u://x")

    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(connector.create_collection("c1", {"dimension": 128}))


def test_milvus_info_resource():
    """Test milvus_info resource function."""
    result = asyncio.run(server.milvus_info())
    assert isinstance(result, str)
    assert "Milvus MCP" in result


def test_main_with_sse(monkeypatch):
    """Test main() with sse=True."""

    class FakeSettingsSSE:
        sse = True

    called = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(server, "get_settings", lambda: FakeSettingsSSE())
    monkeypatch.setattr(server.mcp, "run", fake_run)

    server.main()

    assert "kwargs" in called
    assert called["kwargs"].get("transport") == "sse"


def test_main_without_sse(monkeypatch):
    """Test main() with sse=False."""

    class FakeSettingsNoSSE:
        sse = False

    called = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(server, "get_settings", lambda: FakeSettingsNoSSE())
    monkeypatch.setattr(server.mcp, "run", fake_run)

    server.main()

    assert "kwargs" in called
    # When sse=False, no transport argument
    assert "transport" not in called["kwargs"]


def test_server_lifespan_and_main(monkeypatch):
    # Test lifespan: patch MilvusConnector.from_config to return fake connector
    fake_conn = FakeClient()
    monkeypatch.setattr(server.MilvusConnector, "from_config", classmethod(lambda cls, settings=None: fake_conn))

    # Use the lifespan context manager
    async def run_ctx():
        async with server.server_lifespan(server.mcp) as ctx:
            assert ctx.connector is fake_conn

    asyncio.run(run_ctx())

    # Test main: patch get_settings and mcp.run
    class FakeSettings:
        def __init__(self, sse=False):
            self.sse = sse

    called = {}

    def fake_run(*args, **kwargs):
        called["run_args"] = (args, kwargs)

    monkeypatch.setattr(server, "get_settings", lambda: FakeSettings(sse=True))
    monkeypatch.setattr(server.mcp, "run", fake_run)
    server.main()
    assert "run_args" in called


def test_milvus_connector_error_paths(monkeypatch):
    class ErrorClient(FakeClient):
        def __init__(self, *a, **kw):
            pass

        def list_collections(self):
            raise RuntimeError("boom")

        def describe_collection(self, name):
            raise RuntimeError("boom")

        def search(self, **kwargs):
            raise RuntimeError("boom")

        def query(self, **kwargs):
            raise RuntimeError("boom")

        def hybrid_search(self, **kwargs):
            raise RuntimeError("boom")

        def create_collection(self, **kwargs):
            raise RuntimeError("boom")

        def insert(self, **kwargs):
            raise RuntimeError("boom")

        def delete(self, **kwargs):
            raise RuntimeError("boom")

        def get_collection_stats(self, *a, **kw):
            raise RuntimeError("boom")

        def upsert(self, **kwargs):
            raise RuntimeError("boom")

        def describe_index(self, **kwargs):
            raise RuntimeError("boom")

        def get_load_state(self, *a, **kw):
            raise RuntimeError("boom")

        def list_databases(self):
            raise RuntimeError("boom")

        def load_collection(self, **kwargs):
            raise RuntimeError("boom")

        def release_collection(self, **kwargs):
            raise RuntimeError("boom")

        def create_index(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "MilvusClient", ErrorClient)
    monkeypatch.setattr(server, "IndexParams", FakeIndexParams)
    monkeypatch.setattr(server, "AnnSearchRequest", FakeAnnSearchRequest)
    monkeypatch.setattr(server, "RRFRanker", FakeRRFRanker)
    monkeypatch.setattr(
        server,
        "utility",
        SimpleNamespace(get_query_segment_info=lambda name: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    # Each method should raise ValueError wrapping the original exception
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.list_collections())
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.get_collection_info("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.search_collection("c1", "q"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.query_collection("c1", "id==1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.vector_search("c1", [0.1], "vec"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.hybrid_search("c1", "q", "text", [0.1], "vec", limit=1))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.create_collection("c3", {"dimension": 12}))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.insert_data("c1", [{"id": 1}]))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.delete_entities("c1", "id==1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.get_collection_stats("c1"))
    # sanity check: underlying client method should raise
    conn = server.MilvusConnector(uri="u://x")
    with pytest.raises(RuntimeError):
        conn.client.search(collection_name="c1", data=[])
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.multi_vector_search("c1", [[0.1]], "vec"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.create_index("c1", "vec"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.bulk_insert("c1", {"id": [1]}))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.load_collection("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.release_collection("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.get_query_segment_info("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.upsert_data("c1", {"id": [1]}))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.get_index_info("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.get_collection_loading_progress("c1"))
    with pytest.raises(ValueError):
        conn = server.MilvusConnector(uri="u://x")
        asyncio.run(conn.list_databases())

    # Test use_database: monkeypatch MilvusClient to raise on instantiation
    class BadClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "MilvusClient", BadClient)
    with pytest.raises(ValueError):
        asyncio.run(conn.use_database("db1"))
