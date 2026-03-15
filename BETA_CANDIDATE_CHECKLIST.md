# Flint — Beta Candidate Checklist

This checklist defines when Flint can reasonably be called a beta candidate.

## Core product flow
- [ ] User can install Flint on Linux with `install.sh`
- [x] User can start Flint with `start.sh`
- [x] First-start onboarding appears when no real model is loaded
- [ ] User can upload a `.pth` or `.onnx` model from the UI
- [ ] User can import prior chat history from the UI
- [x] User can export and import runtime state snapshots

## Model handling
- [ ] Active model can be changed from the UI
- [x] Model list shows format, origin, readiness and size
- [x] Official RWKV-7 G1 presets are recommended
- [x] Custom models are allowed at the user's own risk
- [x] Real-world validation against at least one official G1 `.pth` model
- [ ] Real-world validation against at least one RWKV ONNX export

## Training / persistence
- [x] Training profile is visible in the UI
- [x] Snapshot health is visible in the UI
- [x] Reflection and idle reasoning health are visible in the UI
- [x] LoRA last-run status is visible in the UI
- [ ] End-to-end validation of nightly LoRA run on a real local model

## ONNX path
- [x] ONNX backend can classify graph signature
- [x] ONNX backend has a token/state adapter structure
- [x] ONNX backend is covered by tests with a fake session
- [ ] ONNX backend is validated against a real exported RWKV graph
- [ ] Tokenizer strategy is confirmed for the chosen export

## Quality / hardening
- [x] API smoke tests pass
- [x] Upload validation tests pass
- [x] ONNX adapter tests pass
- [ ] More regression coverage for onboarding step transitions
- [ ] More regression coverage for failed model swap rollback

## Packaging
- [x] Linux-first install path exists
- [x] Flatpak scaffold exists
- [ ] Flatpak package tested end-to-end
- [ ] Beta release notes written
