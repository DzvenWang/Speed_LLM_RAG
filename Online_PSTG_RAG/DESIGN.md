# Online_PSTG_RAG v0.1.0 — Causal Recent-Residual RAG for PatchSTG

## One question
Can a frozen strong PatchSTG adapt to the evolving online stream by retrieving recently matured cases, instead of repeatedly changing backbone parameters?

## Core design
PatchSTG is completely frozen. Each online query is predicted once by the frozen model and represented by its pooled pre-head latent state. After the query's complete 12-step future has matured, the case enters a rolling external memory with:

- latent query key;
- time context (TOD/DOW);
- forecast origin;
- frozen-PatchSTG forecast residual `Y_true - Y_base`;
- future-valid mask.

At a later query, the RAG path retrieves Top-K matured recent cases using:

`latent similarity + temporal-context compatibility + recency`.

The retrieved residuals are softmax-weighted and conservatively fused as a correction to the frozen PatchSTG forecast. By default the correction is applied only to directly masked nodes.

## Strict causal rule
For a case with `window_start=s`, history `H=12`, horizon `P=12`:

- forecast origin = `s + H`;
- full future matures at `s + H + P`;
- it may enter RAG memory only when `future_end <= current_origin`.

The true residual is therefore never available at the time its own forecast is produced.

## Why residual memory
Static state-copy retrieval failed because similar historical windows were often not transferable. Online residual memory asks a narrower question: under the *current deployed regime*, how has the frozen PatchSTG recently erred on similar cases? This information did not exist at the offline training cutoff and is therefore genuinely new external memory.

## v0.1 scope
Included:
- frozen PatchSTG reference;
- recent rolling memory;
- latent/context/recency retrieval;
- weighted residual fusion;
- deterministic confidence gate;
- Top-K oracle diagnostic (evaluation only);
- chronological segment analysis;
- full logs and retrieval traces.

Not included yet:
- Qwen/LLM reasoning;
- long-term + recent dual memory;
- backbone online fine-tuning;
- learned fusion/gating.

The first success criterion is empirical: recent RAG should produce a positive realized gain and ideally approach/exceed the existing `Online_PSTG last_block_head` baseline (~1.28% on the current development stream).
