import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocessamento do paper solar panel detection within complex backgrounds...

    """
    reference_value = round(image.mean() + 2 * image.std())
    mask_values = np.where(image > reference_value, True, False)
    _, counts = np.unique(mask_values, return_counts=True)
    data_above_reference_value = (counts[1] / counts.sum()) * 100
    if data_above_reference_value > 5:
        image[mask_values] = reference_value
    return image


def approximate_distance(alpha: float, beta: float, altitude: float) -> float:
    """
    Calculates an approximation of the distance
    between PV modules fixed in a position and
    the drone that's making the thermal inspection.

    Parameters
    ----------
    alpha : float or int
        Tilt angle of PV modules.
    beta : float or int
        Drone's angle in relation to the glass surface
        of PV modules during thermal inspection.
    altitude : float or int
        Drone's altitude relative to a flat terrain.

    Returns
    -------
    float
        Distance between the drone and PV modules.
    """

    return altitude / (np.sin(90 - alpha - beta))
