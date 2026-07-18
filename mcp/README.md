# mindfile-mcp

An [MCP](https://modelcontextprotocol.io) server for the
[mindfile](https://github.com/rogerjs93/mindfile) `.gnos` file format. It lets an MCP client
(Claude Desktop, etc.) **create a mindfile, fold technical files (JSON, text, ...) into it as a
single self-contained container, search and summarize them, and checkpoint + audit** for tamper
detection.

Unlike a typical server wrapper, this embeds the mindfile library in-process (mindfiles are local
files, not a running service). It uses the dependency-free HashEmbedder by default, so no ML model
download is needed.

## Install / run

```bash
uvx --from git+https://github.com/rogerjs93/mindfile.git#subdirectory=mcp mindfile-mcp
```

Or from a clone:

```bash
cd mcp
uv sync
uv run mindfile-mcp
```

## Claude Desktop config

```json
{
  "mcpServers": {
    "mindfile": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/rogerjs93/mindfile.git#subdirectory=mcp", "mindfile-mcp"]
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `create_mindfile(path, modality)` | Create a new `.gnos` file (a self-contained, verifiable database) |
| `add_files(path, files[])` | Fold technical files (JSON, text, ...) into the file as a searchable container |
| `add_texts(path, texts[])` | Add raw text snippets as items |
| `search(path, query_text, k)` | Semantic search over the file's contents |
| `summarize(path)` | Human-readable + structured summary of what the file contains |
| `info(path)` | Counts, header metadata, recent history |
| `checkpoint(path, name)` | Create an immutable save point (content-hashed) |
| `audit(path, deep)` | Verify integrity + chain of custody (catches tampering) |
| `seal_bundle(path, files/from_dir)` | One-shot: fold a set of output files into a .gnos, checkpoint, and audit |

## Bridge: seal another tool's run

`seal_bundle` captures a pile of technical output files (e.g. a
[neuro-orchestrator](https://github.com/rogerjs93/neuro-orchestrator) `process_dataset` run's
group-stats and per-subject JSONs) into one verifiable, searchable `.gnos`. An agent with both
MCP servers connected can chain them:

```
# neuro-orchestrator MCP:
process_dataset("ds004796", ["01","02"], group_column="group")
get_progress()                       # poll until complete
# mindfile MCP:
seal_bundle("run.gnos", from_dir="/path/to/orchestrator/outputs")   # fold + checkpoint + audit
```

The result is a single tamper-evident bundle of the run's outputs, with one summary and a save
point you can `audit` later.

## Example flow

```
create_mindfile("study.gnos", modality="clinical")
add_files("study.gnos", ["metrics.json", "qc.json", "notes.txt"])
summarize("study.gnos")          # one summary of the whole bundle
search("study.gnos", "connectivity")
checkpoint("study.gnos", "v1")   # immutable save point
audit("study.gnos")              # later: confirm nothing was tampered
```

## License

MIT.
