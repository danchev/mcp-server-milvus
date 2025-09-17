![PyPI](https://img.shields.io/pypi/v/mcp-server-milvus)
[![PyPI - Downloads](https://static.pepy.tech/badge/mcp-server-milvus)](https://pepy.tech/project/mcp-server-milvus)
![PyPI - Monthly Downloads](https://static.pepy.tech/badge/mcp-server-milvus/month)


# MCP Server for Milvus

> The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. Whether you're building an AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to connect LLMs with the context they need.

This repository contains a MCP server that provides access to [Milvus](https://milvus.io/) vector database functionality.

**Note:** This repository is a fork of https://github.com/zilliztech/mcp-server-milvus.git with various improvements and added support for `uvx` execution, making it easier to run without installation.

![MCP with Milvus](Claude_mcp+1080.gif)

## Prerequisites

Before using this MCP server, ensure you have:

- Python 3.10 or higher
- A running [Milvus](https://milvus.io/) instance (local or remote)
- [uv](https://github.com/astral-sh/uv) installed (recommended for running the server)
- [npx](https://nodejs.org/en/download/) installed (for testing with MCP Inspector)

## Usage

The recommended way to use this MCP server is to run it directly with `uvx` without installation. This is how both Claude Desktop and Cursor are configured to use it in the examples below.

### Running the Server

You can run the MCP server for Milvus using the following command, replacing `http://localhost:19530` with your Milvus server URI:

```bash
uvx mcp-server-milvus --milvus-uri http://localhost:19530
```
Alternatively, you can set environment variables using a `.env` file in your project directory and then run the server using the following `uvx` command:

```bash
# Create a .env file with your Milvus configuration
cat > .env <<EOF
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
MILVUS_DB=default
EOF

# Run the server with the .env file
uvx --env-file .env mcp-server-milvus
```

### Running Modes

The server supports two running modes: **stdio** (default) and **SSE** (Server-Sent Events).

### Stdio Mode (Default)

- **Description**: Communicates with the client via standard input/output. This is the default mode if no mode is specified.

- Usage:

  ```bash
  uvx mcp-server-milvus@latest --milvus-uri http://localhost:19530
  ```

### SSE Mode

- **Description**: Uses HTTP Server-Sent Events for communication. This mode allows multiple clients to connect via HTTP and is suitable for web-based applications.

- **Usage:**

  ```bash
  uvx mcp-server-milvus@latest --sse --milvus-uri http://localhost:19530
  ```

  - `--sse`: Enables SSE mode.

- **Debugging in SSE Mode:**

  If you want to debug in SSE mode, after starting the SSE service, enter the following command:

  ```bash
  npx @modelcontextprotocol/inspector --transport sse --server-url http://localhost:8000/sse
  ```

  The output will be similar to:

  ```plaintext
  % npx @modelcontextprotocol/inspector --transport sse --server-url http://localhost:8000/sse
  Starting MCP inspector...
  ⚙️ Proxy server listening on port 6277
  🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
  ```

  You can then access the MCP Inspector at `http://127.0.0.1:6274` for testing.

## Supported Applications

This MCP server can be used with various LLM applications that support the Model Context Protocol:

- **Claude Desktop**: Anthropic's desktop application for Claude
- **Cursor**: AI-powered code editor with MCP support
- **Custom MCP clients**: Any application implementing the MCP client specification

## Usage with Claude Desktop

### Configuration for Different Modes

#### SSE Mode Configuration

Follow these steps to configure Claude Desktop for SSE mode:

1. Install Claude Desktop from https://claude.ai/download.
2. Open your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Add the following configuration for SSE mode:

```json
{
  "mcpServers": {
    "milvus-sse": {
      "url": "http://your_sse_host:port/sse",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

4. Restart Claude Desktop to apply the changes.

#### Stdio Mode Configuration

For stdio mode, follow these steps:

1. Install Claude Desktop from https://claude.ai/download.
2. Open your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Add the following configuration for stdio mode:

```json
{
  "mcpServers": {
    "milvus": {
      "command": "uvx",
      "args": [
        "mcp-server-milvus@latest",
        "--milvus-uri",
        "http://localhost:19530"
      ]
    }
  }
}
```

4. Restart Claude Desktop to apply the changes.

## Usage with Cursor

[Cursor also supports MCP](https://docs.cursor.com/context/model-context-protocol) tools. You can integrate your Milvus MCP server with Cursor by following these steps:

### Integration Steps

1. Go to `Cursor Settings` > `Features` > `MCP`
2. Click on the `+ Add New MCP Server` button
3. Fill out the form:

   - **Type**: Select `stdio` (since you're running a command)
   - **Name**: `milvus`
   - **Command**: `uvx mcp-server-milvus@latest --milvus-uri http://127.0.0.1:19530`

   > ⚠️ Note: Use `127.0.0.1` instead of `localhost` to avoid potential DNS resolution issues.

### Option 2: Using Project-specific Configuration (Recommended)

Create a `.cursor/mcp.json` file in your project root:

1. Create the `.cursor` directory in your project root:

   ```bash
   mkdir -p /path/to/your/project/.cursor
   ```

2. Create a `mcp.json` file with the following content:

   ```json
   {
     "mcpServers": {
       "milvus": {
          "command": "uvx",
          "args": [
            "mcp-server-milvus@latest",
            "--milvus-uri",
            "http://127.0.0.1:19530"
          ]
       }
     }
   }
   ```

#### For SSE Mode:

1. Start the service by running the following command:

   ```bash
   uv run mcp-server-milvus --sse --milvus-uri http://your_sse_host
   ```

   > **Note**: Replace `http://your_sse_host` with your actual SSE host address.

2. Once the service is up and running, overwrite the `mcp.json` file with the following content:

   ```json
   {
       "mcpServers": {
         "milvus-sse": {
           "url": "http://your_sse_host:port/sse",
           "disabled": false,
           "autoApprove": []
         }
       }
   }
   ```

### Completing the Integration

After completing the above steps, restart Cursor or reload the window to ensure the configuration takes effect.

## Verifying the Integration

To verify that Cursor has successfully integrated with your Milvus MCP server:

1. Open `Cursor Settings` > `MCP`
2. Check if "milvus" or "milvus-sse" appear in the list（depending on the mode you have chosen）
3. Confirm that the relevant tools are listed (e.g., milvus_list_collections, milvus_vector_search, etc.)
4. If the server is enabled but shows an error, check the Troubleshooting section below

## Available Tools

The server provides the following tools:

### Search and Query Operations

- `milvus_text_search`: Search for documents using full text search

  - Parameters:
    - `collection_name`: Name of collection to search
    - `query_text`: Text to search for
    - `limit`: The maximum number of results to return (default: 5)
    - `output_fields`: Fields to include in results
    - `drop_ratio`: Proportion of low-frequency terms to ignore (0.0-1.0)
- `milvus_vector_search`: Perform vector similarity search on a collection
  - Parameters:
    - `collection_name`: Name of collection to search
    - `vector`: Query vector
    - `vector_field`: Field name for vector search (default: "vector")
    - `limit`: The maximum number of results to return (default: 5)
    - `output_fields`: Fields to include in results
    - `filter_expr`: Filter expression
    - `metric_type`: Distance metric (COSINE, L2, IP) (default: "COSINE")
- `milvus_hybrid_search`: Perform hybrid search on a collection
  - Parameters:
    - `collection_name`: Name of collection to search
    - `query_text`: Text query for search
    - `text_field`: Field name for text search
    - `vector`: Vector of the text query
    - `vector_field`: Field name for vector search
    - `limit`: The maximum number of results to return
    - `output_fields`: Fields to include in results
    - `filter_expr`: Filter expression
- `milvus_query`: Query collection using filter expressions
  - Parameters:
    - `collection_name`: Name of collection to query
    - `filter_expr`: Filter expression (e.g. 'age > 20')
    - `output_fields`: Fields to include in results
    - `limit`: The maximum number of results to return (default: 10)

### Collection Management

- `milvus_list_collections`: List all collections in the database

- `milvus_create_collection`: Create a new collection with specified schema

  - Parameters:
    - `collection_name`: Name for the new collection
    - `collection_schema`: Collection schema definition
    - `index_params`: Optional index parameters

- `milvus_load_collection`: Load a collection into memory for search and query

  - Parameters:
    - `collection_name`: Name of collection to load
    - `replica_number`: Number of replicas (default: 1)

- `milvus_release_collection`: Release a collection from memory
  - Parameters:
    - `collection_name`: Name of collection to release

- `milvus_get_collection_info`: Lists detailed information like schema, properties, collection ID, and other metadata of a specific collection.
  - Parameters:
    - `collection_name`:  Name of the collection to get detailed information about

### Data Operations

- `milvus_insert_data`: Insert data into a collection

  - Parameters:
    - `collection_name`: Name of collection
    - `data`: Dictionary mapping field names to lists of values

- `milvus_delete_entities`: Delete entities from a collection based on filter expression
  - Parameters:
    - `collection_name`: Name of collection
    - `filter_expr`: Filter expression to select entities to delete

## Environment Variables

- `MILVUS_URI`: Milvus server URI (can be set instead of --milvus-uri)
- `MILVUS_TOKEN`: Optional authentication token
- `MILVUS_DB`: Database name (defaults to "default")

## Development

### Testing During Development

To test the server locally using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector uvx mcp-server-milvus@latest
```

## Examples

### Using Claude Desktop

#### Example 1: Listing Collections

```
What are the collections I have in my Milvus DB?
```

Claude will then use MCP to check this information on your Milvus DB.

```
I'll check what collections are available in your Milvus database.

Here are the collections in your Milvus database:

1. rag_demo
2. test
3. chat_messages
4. text_collection
5. image_collection
6. customized_setup
7. streaming_rag_demo
```

#### Example 2: Searching for Documents

```
Find documents in my text_collection that mention "machine learning"
```

Claude will use the full-text search capabilities of Milvus to find relevant documents:

```
I'll search for documents about machine learning in your text_collection.

> View result from milvus-text-search from milvus (local)

Here are the documents I found that mention machine learning:
[Results will appear here based on your actual data]
```

### Using Cursor

#### Example: Creating a Collection

In Cursor, you can ask:

```
Create a new collection called 'articles' in Milvus with fields for title (string), content (string), and a vector field (128 dimensions)
```

Cursor will use the MCP server to execute this operation:

```
I'll create a new collection called 'articles' with the specified fields.

Collection 'articles' has been created successfully with the following schema:
- title: string
- content: string
- vector: float vector[128]
```

## Troubleshooting

### Common Issues

#### Connection Errors

If you see errors like "Failed to connect to Milvus server":

1. Verify your Milvus instance is running: `docker ps` (if using Docker)
2. Check the URI is correct in your configuration
3. Ensure there are no firewall rules blocking the connection
4. Try using `127.0.0.1` instead of `localhost` in the URI

#### Authentication Issues

If you see authentication errors:

1. Verify your `MILVUS_TOKEN` is correct
2. Check if your Milvus instance requires authentication
3. Ensure you have the correct permissions for the operations you're trying to perform

#### Tool Not Found

If the MCP tools don't appear in Claude Desktop or Cursor:

1. Restart the application
2. Check the server logs for any errors
3. Verify the MCP server is running correctly
4. Press the refresh button in the MCP settings (for Cursor)

### Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/zilliztech/mcp-server-milvus/issues) for similar problems
2. Join the [Zilliz Community Discord](https://discord.gg/zilliz) for support
3. File a new issue with detailed information about your problem
