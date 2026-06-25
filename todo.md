# TODO

## Phase 2: Scale & Validate (Current Focus)

### High Priority

- [ ] Create scenario generator for 500+ turn conversations
- [ ] Add 3 scenarios at 500, 750, and 1000 turns
- [ ] Implement GPT-2 tokenizer as offline fallback (no auth needed)
- [ ] Run full benchmark showing 30-50% reduction at default budget
- [ ] Validate with real Mistral-7B answers on Azure

### Medium Priority

- [ ] Add decay factor sweep (0.92, 0.94, 0.96, 0.98) for long conversations
- [ ] Add latency timing to ExperimentRunner (ms per scoring pass)
- [ ] Generate matplotlib charts (token count vs quality per strategy)
- [ ] Add "regression" scenario (contextually important but low cosine similarity)
- [ ] Add scenario with code snippets and structured data

### Lower Priority

- [ ] Set up GitHub Actions CI (run --dummy-llm on push)
- [ ] Add `--output-format json` option to reporter
- [ ] Cache embeddings to disk (avoid recomputing across runs)
- [ ] Pre-bake Azure VM image with NVIDIA drivers (avoid DKMS compile)
- [ ] Tag v0.2.0 release with scale results

## Infrastructure Improvements

- [ ] Use `az vm run-command` in deploy.sh as SSH fallback
- [ ] Add budget alert ($5 threshold) on Azure subscription
- [ ] Create Terraform alternative to shell provisioning scripts
- [ ] Support Azure Container Instances for serverless benchmark

## Code Quality

- [ ] Add unit tests for scoring math (cosine × decay at various ages)
- [ ] Add integration test (stub LLM + deterministic embeddings)
- [ ] Add type annotations and mypy strict mode
- [ ] Parallelize embedding computation for large conversations
- [ ] Add progress bar for long scenarios (tqdm)

## Documentation

- [x] memory.md — current state and achievements
- [x] goals.md — phase objectives with metrics
- [x] ll.md — lessons learned (17 items)
- [x] roadmap.md — 5-phase plan
- [x] BENCHMARK.md — Azure instructions
- [x] results/azure_benchmark_results.md — findings
- [ ] Add CONTRIBUTING.md if opening to team
