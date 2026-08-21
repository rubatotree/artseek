# Security Policy

ArtSeek is a local-first application. The web UI binds to `127.0.0.1` by default and must not listen on public interfaces unless the operator explicitly asks for it. Images, embeddings, and thumbnails stay on the local machine; the only optional network calls are Hugging Face model downloads and Ollama on the configured local URL.

## Reporting a vulnerability

Please report vulnerabilities privately via [GitHub Security Advisories](https://github.com/rubatotree/artseek/security/advisories/new). Do not open a public GitHub issue or discussion for security reports.

Include the affected version or commit, a reproduction, and the impact (for example: path traversal on thumbnails, binding to `0.0.0.0`, or leaking local file paths). You will get an acknowledgement when the report is received.
