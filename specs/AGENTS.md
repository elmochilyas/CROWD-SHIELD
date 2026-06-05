# AGENTS.md — Instructions for AI Coding Agent

## Project: CrowdGuard
Early crowd riot detection for stadium zones — Football Tech Morocco Hackathon

---

## How to use these specs

Each file in `specs/` is the authoritative definition for one module.
Read ALL specs before writing ANY code.

**Reading order:**
1. `SPEC.md` — master overview, data types, architecture
2. `SPEC_config.md` — constants (required before all others)
3. `SPEC_frame_reader.md`
4. `SPEC_detector.md`
5. `SPEC_behavior.md`
6. `SPEC_alert.md`
7. `SPEC_renderer.md`
8. `SPEC_main.md`

---

## Implementation rules

1. **One file per module.** Do not merge modules. Do not add files not in the spec.
2. **All numeric constants come from `config.py`.** Never hardcode thresholds, colors, or model names elsewhere.
3. **Shared types live in `pipeline/__init__.py`.** Import from there everywhere.
4. **Each module's acceptance criteria = its test cases.** After implementing a module, verify each criterion manually or with a short test script.
5. **No print statements except in `main.py`.** Use them only for the console log.
6. **Do not install additional dependencies** beyond `requirements.txt` without flagging it.

---

## Implement in this order

```
Step 1:  pipeline/__init__.py   (data types)
Step 2:  config.py              (constants)
Step 3:  pipeline/frame_reader.py
Step 4:  pipeline/detector.py
Step 5:  pipeline/behavior.py
Step 6:  pipeline/alert.py
Step 7:  pipeline/renderer.py
Step 8:  main.py
```

After each step, the module should be importable without errors before moving to the next.

---

## Known constraints

- YOLOv8 weights are downloaded automatically on first run by Ultralytics.
- ByteTrack is built into Ultralytics — no separate install needed.
- `bytetrack.yaml` is a built-in tracker config in the Ultralytics package.
- On CPU, use `yolov8n.pt` and `yolov8n-pose.pt` (nano) for acceptable fps.
- OpenCV's `cv2.putText` does not render Unicode — use ASCII-safe alternatives for symbol characters (e.g. `[WARN]` instead of `⚠`).

---

## Quick test checklist (run before demo)

- [ ] `python main.py --source test_video.mp4 --no-display` completes without error
- [ ] HUD bar visible in output frames with correct zone label
- [ ] Bounding boxes appear on persons in different colors by risk tier
- [ ] Console prints zone state every 30 frames
- [ ] `--output result.mp4` produces a valid playable video
- [ ] Running on a calm crowd video → zone stays at CALM or WATCH
