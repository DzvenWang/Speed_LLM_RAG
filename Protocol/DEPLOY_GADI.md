# Gadi deployment order

1. Place `Protocol`, `Backbone`, and `Baseline` under `/scratch/mh20/zw1165/Speed_LLM_RAG/`.
2. Run Protocol preflight and build/freeze `development`.
3. Clone/pin Backbone and Baseline official vendor repositories.
4. Prepare Backbone `screening` cache and Baseline `development` cache; both must report the **same Protocol development stage hash**.
5. Submit Backbone four-model screening and Baseline compatibility/development work.
6. Later build Protocol `formal` using all eligible unique Train view-groups (no exact family quota), run full-train formal experiments, freeze configurations, then authorize/build Protocol `evaluation` exactly once.
