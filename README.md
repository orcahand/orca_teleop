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

Gloves always take two programs, on every OS:

- The **bridge** runs on the Linux computer that the gloves' USB dongle is plugged
  into (the "glove computer"). There the MANUS SDK reads the gloves, and the bridge
  sends every frame on a ZMQ address (port 2044).
- **Teleop** reads that address on any OS (Linux, macOS, Windows) and retargets the
  glove's hand with the same retargeter as the webcam.

The two can run on one computer or on two.

```text
gloves -> dongle -> [glove computer] MANUS SDK -> bridge.py --ZMQ :2044--> [teleop computer] orca-teleop-hand --source manus
```

### What you need

- MANUS gloves with their USB dongle, and a MANUS licence with the SDK, SDK
  Integrated and Linux features (the Robotics edition). The licence is on the
  dongle, so keep the dongle plugged in.
- A glove computer with Linux on x86_64 or ARM64 and Python 3.10 or newer. The
  commands below are for Ubuntu/Debian. Windows with MANUS Core is not supported yet.
- A teleop computer with this repo set up (see [Setup](#setup)). It can be the glove
  computer.

### 1. Set up the glove computer (once)

1. Get the MANUS SDK package for Linux from MANUS
   ([SDK getting started](https://docs.manus-meta.com/latest/Plugins/SDK/getting%20started/))
   and unzip it to `~/MANUS_SDK`.
2. Install the SDK's libraries and its udev rules (they let the SDK open the
   dongle; the script uses apt-get), then plug the dongle in again:

   ```bash
   sudo bash ~/MANUS_SDK/C++/SDKMinimalClient/install-dependencies.sh
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

3. Get the bridge. It is only the folder `src/orca_teleop/ingress/manus/` of this
   repo. Clone the repo (`git clone https://github.com/orcahand/orca_teleop.git`), or
   copy the folder over from the teleop computer:
   `scp -r src/orca_teleop/ingress/manus USER@GLOVE_PC:~/manus-bridge`.
4. Make one Python environment for MANUS's tools and the bridge:

   ```bash
   python3 -m venv ~/manus-venv && source ~/manus-venv/bin/activate
   pip install cffi numpy pyzmq "protobuf>=4.25"
   ```

### 2. Calibrate the gloves (once per person)

Each person calibrates the gloves for their own hand. Without it, the glove's hand
does not match the wearer's, and the robot hand is off.

1. In the environment from step 1.4, start MANUS's SDK client:

   ```bash
   source ~/manus-venv/bin/activate && export PYTHONPATH=~/MANUS_SDK/Python
   cd ~/MANUS_SDK/Python && python examples/sdk_client.py
   ```

2. Choose Integrated, then `C` (Glove Calibration): `H` picks the hand, `S` starts,
   `E` runs a step and `P` goes to the next, `F` finishes, and `K` saves
   `left_0x<id>.mcal` or `right_0x<id>.mcal` in the current folder.
3. Rename the files to `Calibration_left.mcal` and `Calibration_right.mcal` (the
   names the bridge loads, and the names the older `manus-client` and its
   CalibrationGUI use), and keep them in one folder, for example `~/gloves`.
4. Close the SDK client: only one program at a time can hold the MANUS SDK
   connection.

### 3. Start the bridge (glove computer)

Switch the gloves on, then:

```bash
source ~/manus-venv/bin/activate && export PYTHONPATH=~/MANUS_SDK/Python
python ~/manus-bridge/bridge.py --calibration-dir ~/gloves
```

(From a clone, the bridge is `src/orca_teleop/ingress/manus/bridge.py`.) Stop the
older `manus-client` first if it runs. When all is well, the bridge logs:

```text
Publishing glove frames on tcp://127.0.0.1:2044
MANUS SDK connected (SDKIntegrated)
MANUS licence: ..., sdk, raw, ..., integrated, ..., linuxTarget, ...
Glove 152439973 is the right hand, 25 nodes
Loaded the right glove's calibration /home/you/gloves/Calibration_right.mcal (Success)
Sent per second: ergonomics 121, landscape 1, raw_device_data 121, right 121
```

`--calibration-dir` is the folder with the `.mcal` files (default: the current
folder). A glove without a file streams uncalibrated, with a warning; the bridge
tries the file again every 5 s, so a file you add later is picked up.

### 4. Start teleop (teleop computer)

```bash
uv run --extra teleop --extra manus --with viser==1.1.1 orca-teleop-hand --source manus --dry-run --viewer
```

`--dry-run --viewer` shows the robot hand in the browser (http://localhost:8098)
without moving the real hand: check that it follows your glove, then drop `--dry-run`
to drive the hand. Add
`--model_path` for your hand as in [Run](#run). Teleop reads `127.0.0.1:2044` by
default; `--glove-address HOST:PORT` (or `tcp://HOST:PORT`) sets another address.

### One computer or two

- **One Linux computer:** run the bridge and teleop on it, as shown.
- **Two computers, SSH tunnel (recommended):** start the bridge as shown (it listens
  on its own computer only). On the teleop computer, run
  `ssh -N -L 2044:127.0.0.1:2044 USER@GLOVE_PC` and keep it open, then start teleop
  as shown.
- **Two computers, network:** start the bridge with `--bind tcp://0.0.0.0:2044`,
  open TCP port 2044 in the glove computer's firewall, and add
  `--glove-address GLOVE_PC:2044` to teleop. The stream has no password or
  encryption, so do this only on a network you trust.

### Warnings

The bridge logs each warning on the glove computer and also sends it to teleop,
which logs it once as `Glove bridge: ...` and once more when it clears.

| Warning | Meaning | What to do |
|---|---|---|
| The MANUS SDK is not connected | The SDK finds no dongle or licence. | Plug the dongle in; check the udev rules (step 1.2) and the licence. |
| The dongle sees no glove | No glove is paired and on. | Switch the glove on; pair it with the dongle. |
| Streaming UNCALIBRATED left/right glove | No `.mcal` file for that side, or the SDK rejected it. | Calibrate (step 2) and put the file in `--calibration-dir`. |
| No data from the left/right glove | A glove stopped sending for 1 s. | Check its battery, range and power switch. |
| ... may not come: your MANUS licence does not include ... | The licence lacks a feature that a data stream needs. | Ask MANUS for the feature; the bridge still sends what the SDK gives. |
| Two gloves report the left/right hand | Two gloves claim one side; only the first goes out. | Pair one glove per side. |
| The bridge falls behind | The glove computer is too slow; old hand frames are skipped. | Close other programs, or use a faster computer. |

On the teleop side, `No glove frames from ... yet` means that nothing arrives: check
that the bridge runs, the address, and the SSH tunnel or firewall.

### Troubleshooting

- **MANUS's SDK aborts at start with `basic_string::_M_construct null not valid`**
  (over SSH, or from a service): the system's language settings are missing. The
  bridge sets them itself. For MANUS's own tools, first run
  `export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 LC_MESSAGES=en_US.UTF-8`.
- **The bridge connects but gets no glove** while the SDK client or `manus-client`
  runs: only one program can hold the MANUS SDK connection. Stop the other one.
- **The bridge stops or restarts while teleop runs:** the robot hand holds its last
  pose, and teleop continues by itself when the bridge is back.
- **The bridge's memory grows** by about 0.7 GB per hour. The MANUS SDK (3.2.1) does
  this by itself once it connects, also without the bridge; the older `manus-client`
  grows faster. Restart the bridge between long sessions (teleop continues by itself).

### Test without gloves

The fake bridge sends a made-up hand that opens and closes
(`--landmarks tests/data/pinch_clip.npz` sends a recorded hand instead):

```bash
uv run --extra teleop --extra manus python -m orca_teleop.ingress.manus.fake_bridge --loop
uv run --extra teleop --extra manus orca-teleop-hand --source manus --dry-run
```

### What the bridge sends (for your own code)

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
readers should skip the messages without hands. Every message also carries the
active warnings (`wire.bridge_warnings()`), and a `status` message without hands
carries them at least once a second.

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
