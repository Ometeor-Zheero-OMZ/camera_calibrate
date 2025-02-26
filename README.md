# Camera Calibrator

environment: Ubuntu (WSL2)

## Setup

Installing dependencies

```bash
sudo apt install -y clang cmake pkg-config libopencv-dev
```

## Commands

Simply, you can execute theses commands:

```bash
cargo run -- --calibrate chessboard
```

```bash
cargo run -- --calibrate symmetric
```

```bash
cargo run -- --calibrate asymmetric
```

```bash
cargo run -- --calibrate charuco
```
