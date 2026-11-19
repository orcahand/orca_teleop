# orca_teleop

Repo for teleoperating the ORCA Hand consisting of an Ingress Source (for example Mediapipe, Apple Vision Pro, Rokoko Gloves, etc.) and a URDF-based Retargeter.

The repository follows a standard `src/` layout:

```text
src/orca_teleop/
tests/
```

## Setup

1. Install with `uv`. The `teleop` extra pulls everything the webcam-to-hand
   pipeline and the 3D viewer need:

   ```bash
   uv sync --extra teleop --extra test
   ```

2. Get the hand models. On v2 the retargeter solves on
   `v2/models/urdf/orcahand_right.urdf` of
   [orcahand_description](https://github.com/orcahand/orcahand_description) `main`,
   including its fingertip tip and pad points:

   ```bash
   git clone git@github.com:orcahand/orcahand_description.git ~/Documents/orcahand_description
   ```

   An existing checkout (the old README used `release/15-10-26`):
   `git -C ~/Documents/orcahand_description fetch && git -C ~/Documents/orcahand_description switch main && git -C ~/Documents/orcahand_description pull`.
   `~/Documents/orcahand_description` is the default location. For another
   location (any folder name), set `ORCAHAND_DESCRIPTION_DIR` to that folder. An
   older checkout fails at start with a message naming what it lacks.

3. Calibrate the hand before the first teleop session, with orca_ui or
   orca_core's `scripts/calibrate.py`. On an uncalibrated hand,
   `orca-teleop-hand` starts the full calibration by itself.

## Run

Check the webcam tracking in the 3D viewer first, with no hand attached:

```bash
uv run --extra teleop orca-teleop-urdf --show-video
```

Then drive the hand. `--model_path` is the path of the hand's `config.yaml`
(orca_core stores the calibration next to it); orca_ui keeps it in
`~/.orca_ui/hands/<hand>/config.yaml`. Only one program can hold the hand's
serial port, so press Disconnect in orca_ui first.

```bash
uv run --extra teleop orca-teleop-hand --model_path ~/.orca_ui/hands/orcahand-right/config.yaml --show-video
```

Useful flags: `--dry-run` runs everything except the hand; `--home` moves the
hand to its neutral pose on connect; `--camera-index N` picks another camera
(often needed on Linux).

`--viewer` also shows the hand model in the browser (http://localhost:8098). White
is the commanded pose, orange the pose the motors report (read 10 times a second,
`--readback-hz`), and a table lists both per joint. Where they part, the physical
hand isn't following. It needs viser:
`uv run --extra teleop --with viser==1.1.1 orca-teleop-hand … --viewer`.
`--log-joints FILE` writes every command and motor read with its time (JSON lines),
for measuring how far the hand lags behind its commands; add `--log-current` to log
each motor's current and velocity with every read too (one more bus read, ~4 ms).

Teleop uses orca_core 0.13, as orca_ui 0.4 does. The log's first line names the
orca_core in use.

If the camera loses your hand, the robot hand holds its pose; when your hand is
seen again it eases over to it in a quarter second, as it does at start from
wherever the hand is. Quitting turns torque off where the hand is, without moving
it first.

## MANUS gloves (Linux)

Gloves always take two programs. The **bridge** runs on the Linux computer that
the gloves' USB dongle is plugged into. There the MANUS SDK reads the gloves, and
the bridge publishes every frame on a ZMQ address (port 2044). **Teleop** reads
that address on any OS and retargets the glove's hand with the same retargeter as
the webcam.

```text
gloves -> dongle -> [glove computer] MANUS SDK -> bridge.py --ZMQ :2044--> [teleop computer] orca-teleop-hand --source manus
```

On the glove computer (Linux x86_64 or ARM64):

1. Get the MANUS SDK package for Linux from MANUS
   ([SDK getting started](https://docs.manus-meta.com/latest/Plugins/SDK/getting%20started/)).
   It needs a MANUS licence with the SDK, SDK Integrated and Linux features (the
   Robotics edition). The licence is on the USB dongle, so keep the dongle plugged in.
2. Install the SDK's libraries and its udev rules (they let the SDK open the
   dongle), then plug the dongle in again:

   ```bash
   sudo bash ~/MANUS_SDK/C++/SDKMinimalClient/install-dependencies.sh
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

3. Calibrate each glove for the person who wears it, with MANUS's SDK client:
   `cd ~/MANUS_SDK/Python && python3 examples/sdk_client.py` (needs `pip install cffi`).
   Choose Integrated, then `C` (Glove Calibration): `H` picks the hand, `S` starts,
   `E` runs a step and `P` goes to the next, `F` finishes, and `K` saves
   `left_0x<id>.mcal` or `right_0x<id>.mcal` in the current folder. Rename them to
   `Calibration_left.mcal` and `Calibration_right.mcal`, the names the bridge loads
   (and the older `manus-client` and its CalibrationGUI use).
4. Start the bridge. It needs only its own folder of this repo, not the rest of
   orca_teleop. Only one program can hold the MANUS SDK connection, so first stop
   the SDK client (and the older `manus-client`, if it runs).

   ```bash
   python3 -m venv ~/manus-bridge && source ~/manus-bridge/bin/activate
   pip install cffi numpy pyzmq "protobuf>=4.25,<6"   # Python 3.10 or newer
   export PYTHONPATH=~/MANUS_SDK/Python
   python orca_teleop/src/orca_teleop/ingress/manus/bridge.py --calibration-dir ~/gloves
   ```

   `--calibration-dir` is the folder with the two `.mcal` files (default: the current
   folder). The bridge loads each file when it finds that side's glove. If a file is
   missing or the SDK rejects it, it warns "Streaming UNCALIBRATED ... glove" and
   goes on. It logs the licence's features, then every 5 s what it sends per second.

On the teleop computer:

```bash
uv run --extra teleop --extra manus orca-teleop-hand --source manus --model_path ~/.orca_ui/hands/orcahand-right/config.yaml
```

`--glove-address HOST:PORT` sets where teleop reads (default `127.0.0.1:2044`;
`tcp://HOST:PORT` works too).
`--dry-run` and `--viewer` work as with the webcam.

One computer or two:

- **One Linux computer:** run both commands on it, as shown.
- **Two computers, SSH tunnel:** keep the bridge as shown (it listens on its own
  computer only). On the teleop computer, run
  `ssh -N -L 2044:127.0.0.1:2044 USER@GLOVE_PC`, then teleop as shown.
- **Two computers, network:** start the bridge with `--bind tcp://0.0.0.0:2044`,
  open TCP port 2044 in the glove computer's firewall, and add
  `--glove-address GLOVE_PC:2044` to teleop. The stream has no password or
  encryption, so do this only on a network you trust.

If MANUS's SDK aborts at start with `basic_string::_M_construct null not valid` (over
SSH, from a service), locale variables are missing: the bridge sets them itself; for
MANUS's own tools, first run
`export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 LC_MESSAGES=en_US.UTF-8`.

Without gloves, the fake bridge sends a made-up hand that opens and closes
(`--landmarks tests/data/pinch_clip.npz` sends a recorded hand instead):

```bash
uv run --extra teleop --extra manus python -m orca_teleop.ingress.manus.fake_bridge --loop
uv run --extra teleop --extra manus orca-teleop-hand --source manus --dry-run
```

Each hand goes out as 25 skeleton nodes (the wrist, then each finger's joints) in
metres, sorted by the SDK's own node labels; the retargeter uses 21 of them.
`src/orca_teleop/ingress/manus/conversion.py` holds that contract, and
`tests/test_manus.py` checks it. The older C++ client
([manus-client](https://github.com/orcahand/manus-client), `manus-client run`)
sends the same hand messages on `127.0.0.1:2044`, so teleop reads it the same way.

The bridge also sends everything else the MANUS SDK gives in integrated mode, one
message per SDK callback on the same address: the ergonomics joint angles (degrees),
the raw device (fingertip sensor) data, the landscape (gloves, dongle, battery,
signal, licence) and system messages. Each message names its stream in `sdk_stream`
and holds the SDK's own structs as JSON in `sdk_json` (field names as in MANUS's
`ManusSDKTypes.h`; `wire.sdk_data()` reads them). Teleop uses only the hands; other
readers should skip the messages without hands. If the licence lacks a stream's
feature, the bridge says so and leaves that stream out (never the hands). Its warnings
(the SDK not connected, a feature the licence lacks, an UNCALIBRATED glove, a lost
glove) go to teleop in every message, and in a `status` message without hands at least
once a second; teleop logs each once when it appears and once when it clears.

## Retargeting

The default teleop backend is `adaptive_analytical`, an Orca-native port of the Wuji-style
analytical retargeting strategy. It uses the full MediaPipe hand pose, explicit Orca frame
mappings from YAML, Pinocchio forward kinematics/Jacobians, and `nlopt` for bounded
per-frame optimization.

Each frame it matches the fingertips and the direction of every finger segment to
yours, closes a pinch when your thumb and a fingertip come within about 1.5 cm, and
smooths the result with a One Euro filter (steady when you hold still, quick when you
move).

It loads `src/orca_teleop/retargeting/configs/adaptive_analytical_orca_tuned.yaml` by
default. Pass `--retarget-config path/to/config.yaml` to experiment with frame maps or
weights. `--retargeter rmsprop` selects the legacy fingertip key-vector retargeter for
comparison.

## Measure fidelity

Record a webcam session with `--record DIR`. Replay it into the normal pipeline, as
if it came from the camera, or score how well each finger is matched. The replay
needs the robot side running first, waiting for a publisher:
`orca-teleop-urdf --no-local`, or `orca-teleop-hand … --no-local`.

```bash
uv run --no-sync python -m orca_teleop.ingress.replay_publisher recordings/thumb-01 --loop
uv run --no-sync python scripts/fidelity_inspector.py recordings/thumb-01 \
    --model-path ~/.orca_ui/hands/orcahand-right/config.yaml --no-viewer
```

`tests/test_fidelity.py` replays a recorded pinch clip and fails if pinch, curl,
jitter or lag get worse. It needs the v2.1 model (`ORCAHAND_DESCRIPTION_DIR`) and
skips without it; this repo has no CI, so run it locally.

Run the test suite from the repository root with:

```bash
uv run --extra teleop --extra test pytest
```

## Record Quest WebXR demonstrations on a physical OrcaHand

The dataset recorder includes the complete Quest ingress: it serves its own
WebXR page, receives 25 hand joints from Quest Browser over a WebSocket, maps
them to the same 21-point layout used by the original in-hand MuJoCo
demonstrations, retargets them, and drives the physical hand.

Start by installing the recording stack (cameras, Quest WebXR, adaptive
retargeter, and LeRobot):

```bash
uv sync --extra learning
```

First validate the motion with the simulated sink:

```bash
uv run python scripts/record_dataset.py \
  --repo-id "$HF_USERNAME/orca-quest-safety-check" \
  --task "check WebXR retargeting" \
  --backend sim \
  --source metaquest \
  --local \
  --hand right \
  --episode-end space \
  --num-episodes 1
```

For the real hand, run:

```bash
uv run python scripts/record_dataset.py \
  --repo-id "$HF_USERNAME/orca-quest-hardware" \
  --task "pick up the block" \
  --backend hardware \
  --source metaquest \
  --local \
  --hand right \
  --episode-end space \
  --num-episodes 20 \
  --camera front:0
```

Hold the hand still while the retargeter calibrates. Press Space to save the
current episode and begin the next; press `q` or Escape to finish. The command
records measured hardware joint positions as `observation.state`, commanded
joint positions as `action`, and each configured camera as
`observation.images.<name>`.

The command hosts the Quest page on port `8765`. WebXR requires a secure
context, so expose it in a second terminal:

```bash
ngrok http 8765
```

Open the resulting HTTPS URL in Quest Browser and tap **Start hand tracking**.
For a wired setup, use:

```bash
adb reverse tcp:8765 tcp:8765
```

Then open `http://localhost:8765` in Quest Browser; browsers treat localhost as
a secure context.

For a split-machine setup, omit `--local` on the robot and run the publisher on
the Quest-side machine:

```bash
uv run python -m orca_teleop.ingress.metaquest.publisher \
  --server ROBOT_IP:50051 \
  --hand right
```

The publisher machine hosts the WebXR page and forwards the reduced landmarks
to the robot over gRPC. Keep `--source metaquest` on the recorder so it treats
the incoming frames as the WebXR convention. The live adapter deliberately
does not apply the legacy HTS/Unity handedness flip.
