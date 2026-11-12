KEY_VECTORS_SHAPE = (5, 3)


# Where MediaPipe's TIP landmark sits relative to the model's fingertip frame
# ({side}_{digit}_tip: on the skin at the end of the fingertip, Z out of the skin):
# a few millimetres inside the finger, i.e. along the frame's -Z.
TIP_LANDMARK_OFFSET_M: tuple[float, float, float] = (0.0, 0.0, -0.003)

# Auto-scale calibration: number of frames to collect MANO key-vector magnitudes
# before computing the MANO→metres scale ratio.
CALIBRATION_FRAMES = 30

# Per-finger loss weights. Equal coefficients only rescale the loss (and thus
# the gradient)
DEFAULT_LOSS_COEFFS: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)

# Translation applied after rotating normalized MANO joints into the URDF base
MANO_TO_URDF_TRANSLATION = (0.0, 0.0, -0.02)
