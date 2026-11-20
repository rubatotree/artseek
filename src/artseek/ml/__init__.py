"""ML embedder, tagger, captioner, LLM, and indexing pipeline.

Submodules are imported explicitly (`artseek.ml.embedder`, `artseek.ml.llm`, …)
so the web API can import the service layer without eagerly loading torch.
"""
