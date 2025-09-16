"""Milvus MCP Server."""

from importlib.metadata import version

try:
    __version__ = version("mcp-server-milvus")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
