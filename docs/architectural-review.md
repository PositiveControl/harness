# Architectural Review: Harness Project

**Date:** April 28, 2026  
**Role:** SOTA LLM Harness Architect  
**Status:** Initial Review

## Executive Summary

The `harness` project is a highly sophisticated, local-first agentic system that demonstrates state-of-the-art (SOTA) patterns in local LLM optimization, memory management, and robust tool orchestration. The architecture is modular, avoiding common "agent spaghetti" traps through clean abstractions and a tiered information lifecycle.

---

## 1. Design & Structure
**Technical Maturity: High**

- **Identity-as-Configuration:** Decoupling persona, voice, and memories into a declarative `character/` directory is an excellent design choice. It enables multi-agent experimentation and personality hot-swapping without touching the core runtime.
- **Adapter-Centric Model Layer:** The `harness.model` layer cleanly abstracts inference runtimes (MLX, Ollama). The MLX implementation is particularly advanced, supporting **speculative decoding** (draft models) and **LoRA weight merging** at load time.
- **Dynamic Tool Profiles:** Using named tool-set profiles (`minimal`, `coding`, `memory`) is a smart token-management strategy. It keeps the system prompt focused and minimizes the "schema overhead" (limited to ~1,500 tokens), which is critical for local models with smaller context windows or slower throughput.

## 2. Data Flow
**Technical Maturity: High**

- **Tiered Memory Lifecycle:** The `seed` (immutable) → `working` (extracted) → `consolidated` (merged/promoted) pipeline is a robust implementation of the information lifecycle. Using a **Scribe** to extract memories from transcripts ensures that the memory is semantic and attributed, rather than just raw text.
- **Hybrid Retrieval (RAG Best Practices):** The system uses **Reciprocal Rank Fusion (RRF)** to combine dense-cosine embeddings with BM25 (SQLite FTS5). The use of **structured tag headers** (e.g., `[tier: X; date: Y]`) injected into the embedding text is a clever trick to give dense retrieval "structured awareness."
- **Two-Pass Persona Pipeline:** Separating "Substance" (reasoning/facts) from "Voice" (persona rewrite) is the industry standard for high-fidelity agents. It allows the reasoning model to be "boring and correct" while the rewriter handles the stylistic nuances.

## 3. Security
**Technical Maturity: Medium**

- **Authorization Tiers:** The distinction between `read` and `write` tool tiers, with a mandatory confirmation loop for `write` actions, is a foundational safety pattern.
- **Path-Based Sandboxing:** Filesystem tools use `Path.resolve()` and `relative_to` checks to prevent directory traversal.
- **Identified Risks:**
    - **Execution Isolation:** The `ShellTool` runs directly on the host OS. While gated by user confirmation, it lacks **OS-level isolation** (e.g., Docker, nsjail, or Firecracker).
    - **Credential Management:** A centralized secret management pattern for third-party tool keys (e.g., SearchWeb) is not explicitly formalized in the core architecture.

## 4. SOTA Recommendations

### Advanced Observability
Implement structured tracing (e.g., OpenTelemetry). Adding **Trace-IDs** that follow a request through the router, tool loop, and memory retrieval would significantly aid in debugging "reasoning drifts."

### Execution Sandboxing
Move the `shell` and `write_file` tools into a lightweight sandbox (e.g., **Firecracker**, **nsjail**, or **WebAssembly**) to allow for safer autonomy.

### Multimodal Integration
Add **Vision (VLM)** support (e.g., Qwen2-VL) to allow the agent to "see" the user's workspace, leveraging MLX's local efficiency.

### Agent Initiative
Transition from a strictly request-response model to an **async, event-driven loop**. Implement "Active Monitors" that can trigger agent actions based on workspace events.

### Streaming Citations
Enhance the UI with **inline streaming citations**, showing the user exactly which memory or fact supported a specific response segment in real-time.

---

**Final Verdict:** 
`harness` is a premier example of local-first agent engineering. Strengthening **execution isolation** and **structured observability** would make it the gold standard for secure, autonomous local AI.
