###########################################################################################
# File to export an refractive index 2D landscape into a (binary) gds file for fabrication.
###########################################################################################

import numpy as np
import os
from datetime import datetime

from src.GPE_solver.design_utils import generate_directional_coupler, mask_to_closed_waveguides_gds

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
WRITE_DIR = os.path.join(FILE_DIR, "designs", TIMESTAMP)

if __name__ == "__main__":    


    wg_width = 0.5
    wg_separation_sides = wg_width + 2.
    wg_separation_middle_vals = [wg_width, wg_width + 0.5]
    coupler_length_vals = [25]
    taper_length = 30 # length of taper to coupler
    taper_bend = 5 # bend, curvature of taper
    eps = 1e-5

    x = np.linspace(0, 2 * taper_length + 2 * coupler_length_vals[0], 4001) # in um
    y = np.linspace(-5, 5, 1001)

    # create write dir
    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    for wg_separation_middle in wg_separation_middle_vals:
        for coupler_length in coupler_length_vals:

            # create coupler
            profile_coupler = generate_directional_coupler(
                x=x, y=y,
                width=wg_width, 
                separation_sides=wg_separation_sides, 
                separation_middle=wg_separation_middle + eps,
                coupler_length=coupler_length, 
                taper_length=taper_length, 
                bend=taper_bend
            )

            # write file
            filename = os.path.join(WRITE_DIR, f"coupler__length_{coupler_length}_gap_{wg_separation_middle - wg_width}.gds")
            mask_to_closed_waveguides_gds(profile_coupler, x=x, y=y, gds_filename=filename)