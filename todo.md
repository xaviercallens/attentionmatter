# TODO

## All Phases Complete

- [x] Phase 1: PoC Validation (v0.1.0)
- [x] Phase 2: Scale & Validate (46-50% reduction)
- [x] Phase 3: Production Integration (standalone module)
- [x] Phase 4: Advanced Scoring (classifier, cross-encoder, ALiBi)
- [x] Phase 5: Scale & Generalize (multi-modal, sharing, streaming, feedback)
- [x] Post-v1.0: Tests, CI, logging, py.typed

## Remaining (Nice-to-Have)

- [ ] Increase test coverage to >90% (current: core paths covered)
- [ ] Add property-based tests with Hypothesis
- [ ] Add Prometheus-style metrics export
- [ ] Async embedding support for OpenAI backend
- [ ] Disk-backed embedding cache with TTL
- [ ] Terraform alternative to shell provisioning scripts
- [ ] Benchmark on 128k+ context models
- [ ] Train classifier on real conversation data
- [ ] Neural cross-encoder evaluation (requires model download)
- [ ] Publish to PyPI as `attn-scorer`

## Known Issues

- Corporate SSL proxy blocks model downloads (workaround: `SSL_CERT_FILE=""`)
- SSH to Azure VMs blocked by corporate proxy (workaround: `az vm run-command`)
- Positional bias training needs small learning rate for large age values
- DummyLLMClient regex doesn't catch all fact patterns (e.g., "penicillin")
