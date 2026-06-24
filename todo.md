# TODO

## Immediate Priority

- [ ] Initialize git repo and push to GitHub
- [ ] Run `./azure/backup.sh` to create initial archive
- [ ] Provision Azure VM: `./azure/provision.sh`
- [ ] Execute full GPU benchmark: `./azure/deploy.sh docker`
- [ ] Download and review results
- [ ] Teardown Azure resources: `./azure/teardown.sh`

## Post-Benchmark

- [ ] Analyze decay factor impact across scenarios
- [ ] Generate bar charts (token count vs quality per strategy)
- [ ] Write summary findings document
- [ ] Tag v0.1.0 release on GitHub
- [ ] Archive final results: `./azure/backup.sh`

## Code Improvements

- [ ] Add unit tests for scoring math (cosine × decay at various ages)
- [ ] Add integration test with stub LLM verifying selection behavior
- [ ] Cache embeddings to disk (avoid recomputing across runs)
- [ ] Support loading config from `azure/benchmark-config.json` via CLI
- [ ] Add `--output-format json` option to reporter
- [ ] Parallelize embedding computation for large conversations

## Scenarios

- [ ] Add scenario with 500+ turns to stress-test at scale
- [ ] Add scenario where query is ambiguous (tests scorer robustness)
- [ ] Add scenario with multiple sessions (full cross-session memory test)
- [ ] Add scenario with code/structured data in context

## Infrastructure

- [ ] Add GitHub Actions CI (run `--dummy-llm` on every push)
- [ ] Create Terraform alternative to shell provisioning scripts
- [ ] Add cost monitoring/alerting for Azure VM uptime
- [ ] Support Azure Container Instances for serverless benchmark runs
