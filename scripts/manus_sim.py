"""Run MANUS glove teleop in sim. Start the glove bridge (ingress/manus/bridge.py) first."""

from orca_teleop.pipeline import run_manus_local
from orca_teleop.sim import OrcaHandSimSink

if __name__ == "__main__":  # the publisher is a spawned child, which imports this file again
    run_manus_local(sink=OrcaHandSimSink(env_name="right", render_mode="human"))
