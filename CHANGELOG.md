# CHANGELOG


## v0.1.0 (2025-05-17)

### Bug Fixes

- Modify vector_search signature ([#24](https://github.com/danchev/mcp-server-milvus/pull/24),
  [`2a4e323`](https://github.com/danchev/mcp-server-milvus/commit/2a4e323993ac306737136bbfc41773106d148a8d))

Co-authored-by: ji-young-shin <ji-young.shin@navercorp.com>

### Chores

- Configure semantic-release publishing options
  ([`ae3e72c`](https://github.com/danchev/mcp-server-milvus/commit/ae3e72c64a0f102ba5edee43e88391a47b85c5c7))

### Continuous Integration

- Add pre-commit config
  ([`f909275`](https://github.com/danchev/mcp-server-milvus/commit/f909275c59d3cc26dae1eeb0dde2b835e6741e2d))

### Features

- Add CLI argument parsing with env fallbacks
  ([`bf0a3fb`](https://github.com/danchev/mcp-server-milvus/commit/bf0a3fbc8064b5bed9a7a54cbc73d8ba4df698d4))

- Add argument parser for --milvus-uri, --milvus-token, and --milvus-db - Implement environment
  variable fallbacks for CLI arguments - Add setup_environment function to configure env from args -
  Update main to handle arguments before server startup
