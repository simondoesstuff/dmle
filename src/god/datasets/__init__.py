from god.datasets.base import Dataset as Dataset
from god.datasets.mandelbrot import (
    ESCAPE_RADIUS as ESCAPE_RADIUS,
    K_DEFAULT as K_DEFAULT,
    SCALE as SCALE,
    encode as encode,
    enc_dim as enc_dim,
    decode as decode,
    decode_magnitude as decode_magnitude,
    make_mandelbrot_dataset as make_mandelbrot_dataset,
    mandelbrot_mag as mandelbrot_mag,
)
from god.datasets.modular import (
    ModularConfig as ModularConfig,
    Op as Op,
    input_dim as input_dim,
    make_modular_dataset as make_modular_dataset,
)
