# Environment Report — WayFinder / IDR System
**Generated:** 2026-09-11
**Phase:** 0 — Environment Discovery

---

## System

| Property | Value |
|----------|-------|
| OS | macOS 26.5.2 (darwin-arm64) |
| Architecture | Apple Silicon ARM64 |
| RAM | 8 GB |
| Disk Free | ~23 GiB |
| Python | 3.14.4 (/opt/homebrew/bin/python3) |
| pip | 26.0.1 |

## GPU / Accelerator

| Property | Value |
|----------|-------|
| GPU | Apple Silicon MPS (Metal Performance Shaders) |
| CUDA | Not available |
| Note | PyTorch will use MPS backend on Apple Silicon |

## Python Packages (Phase 0)

| Package | Version | Status |
|---------|---------|--------|
| numpy | 2.5.3 | OK |
| pandas | 3.0.5 | OK |
| scipy | latest | OK |
| matplotlib | latest | OK |
| scikit-learn | latest | OK |
| filterpy | latest | OK (EKF/UKF) |
| torch | NOT YET | Deferred to Phase 5 |
| onnx / onnxruntime | NOT YET | Deferred to Phase 15 |

## Mobile / Android Tooling

| Tool | Version | Status |
|------|---------|--------|
| Flutter | 3.44.1 (stable channel) | OK |
| Dart | 3.12.1 | OK |
| Android SDK | 36.1.0 | OK |
| ADB | 1.0.41 | OK |
| Java | 25.0.2 LTS | OK |
| Xcode | Incomplete | iOS UNAVAILABLE — Android only |
| Connected Devices | 2 available | Android deployment possible |

## Dataset — IO-VNBD

| Property | Status |
|----------|--------|
| ZIP in Downloads | FOUND (IO-VNBD-master.zip, 1.3 MB — LFS pointers only) |
| git-lfs | Installed via Homebrew during Phase 0 |
| Full Clone | IN PROGRESS (git lfs clone from github.com/onyekpeu/IO-VNBD) |
| Expected Size | ~1.3 GB |

### Discovered Structure (from ZIP + GitHub)
```
IO-VNBD-master/
├── Synchronised V abd S datasets/
│   └── Categorised IOVNB Dataset/
│       ├── S (Driver A)/     S1, S2, S3a, S3b, S3c, S4
│       ├── M (Driver B)/
│       ├── Vf (Driver E)/    Vfa01, Vfa02
│       └── Vta (Driver E)/   Vta01a..Vta16+
└── Unsynchronised V and S Dataset/
    └── Uncategorised IOVNB (V and S) Dataset/
        ├── V-Dataset/  V-Vw1..9, V-Y1, V-Y2
        └── S-Dataset/
```

File convention:
- V-*.csv = Vehicle ECU/OBD data
- S-*.csv = Smartphone sensor data (IMU + GNSS)
- Synchronised pairs: same sequence name

Known from literature (unverified until actual CSV read):
- Sampling rate ~10 Hz
- Covers UK, Nigeria, France
- ~40h vehicle data (~1300 km), ~58h smartphone (~4400 km)

## Development Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Navigation Engine | Python 3.14 | Prototyping speed |
| Classical INS | NumPy + SciPy | Lightweight |
| EKF / UKF | filterpy | Mature, well-tested |
| AI Models | PyTorch (deferred) | MPS on Apple Silicon |
| Mobile App | Flutter (Android) | Toolchain available |
| Map Matching | OSM + custom HMM | Offline, no API keys |

## Phase 0 Exit Criteria

- [x] Environment understood
- [x] Python core packages installed
- [x] Flutter/Android toolchain confirmed
- [x] Dataset located (LFS clone in progress)
- [x] Development stack selected
- [x] No unnecessary large dependencies installed
- [ ] Actual CSV columns verified — PENDING (LFS clone running)

**STATUS: IN PROGRESS — Awaiting LFS data download**
