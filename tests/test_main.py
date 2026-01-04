"""Test for __main__.py to ensure entrypoint is covered."""

import runpy
import sys
import types


def test_main_entrypoint(monkeypatch):
    called = {}

    def fake_main():
        called["main"] = True

    sys.modules["mcp_server_milvus.server"] = types.SimpleNamespace(main=fake_main)
    # Run the module as __main__ so __name__ == "__main__"
    runpy.run_module("mcp_server_milvus", run_name="__main__")
    assert called.get("main") is True
