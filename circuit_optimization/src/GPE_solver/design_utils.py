import numpy as np


def generate_bend(x: np.ndarray, y: np.ndarray, y0: float, yend: float, w: float, b: float):
    """
    Generates a binary 2D mask of a waveguide defined by a smooth sigmoid bend.

    Parameters:
    x, y: 1D numpy arrays representing the pixel coordinates.
    y0: Starting y position of the waveguide center.
    yend: Ending y position of the waveguide center.
    w: Width of the waveguide.
    b: Bending parameter (controls steepness of sigmoid).

    Returns:
    A 2D numpy array with 1s representing the waveguide and 0s elsewhere.
    """
    X, Y = np.meshgrid(x, y)
    
    # Define center line of the waveguide
    x_mid = (x[-1] + x[0]) / 2
    x_shift = w * np.sin(np.arctan(((yend - y0)/(4 * b)))) / 2

    center_y_top = y0 + (yend - y0) / (1 + np.exp(- (X - x_mid + x_shift) / b))
    center_y_bottom = y0 + (yend - y0) / (1 + np.exp(- (X - x_mid - x_shift) / b))

    # Define top and bottom edges
    y_top = center_y_top + w / 2
    y_bottom = center_y_bottom - w / 2

    # Generate the mask
    waveguide_mask = ((Y >= y_bottom) & (Y <= y_top)).astype(np.uint8)
    
    return waveguide_mask


def generate_directional_coupler(
        x: np.ndarray, 
        y: np.ndarray, 
        width: float,
        taper_length: float, 
        separation_middle: float, 
        separation_sides: float,
        bend: float,
        coupler_length: None|float = None
        ) -> np.ndarray:
    
    """
    Generate a directional coupler profile (binary) on a rectangular grid (x,y).

    Parameters
    ----------
    x: np.ndarray
        The x-values (direction of photon propagation)
    y: np.ndarray
        The y-values (transverse direction of photon propagation)
    width: float
        The width of the waveguides

    taper_length: float
        The length of the taper.

    separation_middle: float
        The separation between the waveguides in the middle (coupler).

    separation_sides: float
        The separation between the waveguides for incoming and outgoing (edges x0 and x=L).

    bend: float
        The curvature parameter of the sigmoid function for the bend, 
         np.sin(np.arctan(((yend - y0)/(4 * b)))) / 2 
         -- if bend low, very sharp / if bend high, very smooth

    coupler_length: None|float (default None)
        The length of the coupler. Default None, the remaining length in interval x \in [0,L]
        is given to coupler, after subtraction tapers
    """
    
    # define a few length scales
    x_mid, y_mid = (x[-1] + x[0]) / 2., (y[-1] + y[0]) / 2.
    xL, yL = x[-1] - x[0], y[-1] - y[0]
    nx, ny = x.size, y.size

    # default coupler length is total length - 2 * taper
    if coupler_length is None:
        coupler_length = xL - 2 * taper_length
    
    side_length = 0.5 * (xL - coupler_length - 2 * taper_length)

    # select indices tapers
    taper_inds = (x < (x[0] + taper_length + side_length)) & (x >= (x[0] + side_length))
    side_connect_inds = x < (x[0] + side_length)
    n_taper = taper_inds.sum()
    n_side = side_connect_inds.sum()

    # create side_connector
    if n_side > 0:
        side_in = np.zeros((ny, n_side))
        side_in[(y <= y_mid + separation_sides/2 + width/2) & (y >= y_mid + separation_sides/2 - width/2)] = 1
        side_in = side_in + np.flip(side_in, axis=0)
        side_out = np.flip(side_in, axis=1)
    else:
        side_in, side_out = None, None


    # create taper
    taper_up = generate_bend(
        x[taper_inds], y, 
        y0=y_mid + separation_sides/2, 
        yend=y_mid + separation_middle/2, 
        w=width,
        b=bend
    )

    taper_in = taper_up + np.flip(taper_up, axis=0)
    taper_out = np.flip(taper_in, axis=1)

    # create middle connector
    coupler = np.zeros((ny, nx - 2 * n_taper - 2 * n_side))
    coupler[(y <= y_mid + separation_middle/2 + width/2) & (y >= y_mid + separation_middle/2 - width/2)] = 1
    coupler = coupler + np.flip(coupler, axis=0)

    # attach all together
    if n_side > 0:
        full_coupler = np.concatenate([side_in, taper_in, coupler, taper_out, side_out], axis=1)
    else:
        full_coupler = np.concatenate([taper_in, coupler, taper_out], axis=1)

    return full_coupler


def mask_to_closed_waveguides_gds(mask, x, y, layer=1, datatype=0, gds_filename="waveguides_closed.gds"):
    
    """
    Converts a binary mask into a GDS file with closed polygons for each waveguide.

    Assumes each waveguide has two contours (top & bottom), and all contours are clean and do not cross.
    Contours are paired by sorting on the left-most x index (works for left-to-right waveguides).
    """

    # get some relevant packages
    import gdspy
    from skimage import measure

    contours = measure.find_contours(mask, 0.5)

    if len(contours) % 2 != 0:
        raise ValueError(f"Expected even number of contours (top/bottom pairs), found {len(contours)}")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Convert contours to physical coordinates + sort key
    physical_contours = []
    for contour in contours:
        poly_pts = [(x[0] + x_idx * dx, y[0] + y_idx * dy) for y_idx, x_idx in contour]
        leftmost_x = min([pt[0] for pt in poly_pts])  # sort key: leftmost x
        physical_contours.append((leftmost_x, poly_pts))

    # Sort by leftmost x
    physical_contours.sort(key=lambda item: item[0])

    # Drop sort key
    physical_contours = [pts for _, pts in physical_contours]

    # Make or reset cell
    cell_name = "WAVEGUIDES_CLOSED"
    if cell_name in gdspy.current_library.cells:
        del gdspy.current_library.cells[cell_name]
    gds_cell = gdspy.Cell(cell_name)

    # Pair and write polygons
    for i in range(0, len(physical_contours), 2):
        wg = np.concatenate([physical_contours[i], physical_contours[i+1], physical_contours[i][:1]], axis=0)
        gds_cell.add(gdspy.Polygon(wg, layer=layer, datatype=datatype))

    gdspy.write_gds(gds_filename, unit=1e-6, precision=1e-9)
    print(f"Closed GDS file written with {len(physical_contours)//2} waveguides: {gds_filename}")
