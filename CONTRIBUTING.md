# Contributing to Flint

Thanks for your interest in contributing. This is a focused project — a local, private companion bot built on RWKV. Contributions that serve that purpose are welcome.

## What this project is

A local-first companion with:
- Persistent RWKV state as relationship memory
- Automatic nightly LoRA fine-tuning
- Terminal and web UI
- No cloud dependency, no RAG, no tool-calling

## Development setup

```bash
git clone https://github.com/yourname/flint
cd flint
bash install.sh
cp config.example.py config_local.py
# Edit config_local.py with your settings
```

## Areas where contributions are welcome

**Bug fixes** — always welcome, especially if you've tested against real hardware.

**RWKV-PEFT integration** — the `_run_peft_training()` method in `lora/pipeline.py` calls RWKV-PEFT via subprocess with assumed CLI argument names. These may drift across RWKV-PEFT versions. PRs that validate and fix this are very useful.

**Parser improvements** — `tools/parser.py` supports ChatGPT and Claude exports. PRs adding other formats (WhatsApp, Discord, etc.) are welcome, following the same normalized session dict format.

**Performance** — generation speed improvements that don't sacrifice state correctness.

## What to avoid

- RAG or vector database integration — this project intentionally relies on RWKV state
- Cloud sync features — local-first is a core constraint
- Multi-user support — this is a personal companion, not a service
- Heavy new dependencies without strong justification

## Pull request guidelines

- One concern per PR
- Describe what bug you're fixing or what use case you're enabling
- If you change `lora/pipeline.py`, describe what RWKV-PEFT version you tested against
- No breaking changes to the SQLite schema without a migration path

## Reporting issues

Please include:
- OS and Python version
- GPU/VRAM if relevant
- RWKV model variant and strategy string
- Full error traceback
