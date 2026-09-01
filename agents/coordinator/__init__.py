"""Coordinator: fuses specialist findings into a ranked, explainable case file.

The only component that talks to the local LLM (see config.OLLAMA_MODEL). It
narrates evidence that the specialists already computed -- it must never be the
thing that decides whether something is fraud, because LLM output is not
reproducible enough to defend under E5.
"""
