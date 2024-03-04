import dataclasses as dt
from typing import List
import numpy as np


@dt.dataclass
class SimulationParams:
    wavelength: float = 1.3776e-4  # microns, for 9 keV
    det_pix_size: float = 55  # microns, for Merlin
    det_dist = 0.35e6  # detector to sample distance
    npix_det: int = 150  # square detector
    HKL_list: List[int] = dt.field(default_factory=lambda: np.array([[1, 0, 0],
                                                                     [1, 1, 0],
                                                                     [1, 2, 0],
                                                                     [2, 1, 0]]))
    magnitudes_scaling_per_peak: List[int] = dt.field(default_factory=
                                                      lambda: np.array([0.04, 0.035, 0.021, 0.01]) / 2)
    random_scaled_magnitudes: bool = True
    magnitudes_max: float = 0.5
    n_scan_positions: int = 9
    npix_scan_shift: int = 6
    sample_pix_size: float = dt.field(init=False)
    probes_matlab_h5_file: str = "/home/skandel/code/mbptycho/experiments/matlab/datasets_0821/probes.h5"
    poisson_noise: bool = True
    poisson_noise_level: float = None  # not implemented yet
    probe_abs_max: float = 728 # Reading this from the matlab data before normalization


    def __post_init__(self):
        # The starting voxelation could be anything really, as long as we are interpolating it to get the correct
        # pixellation before projection to the detector.
        self.sample_pix_size = self.det_dist * self.wavelength / (self.npix_det * self.det_pix_size)


@dt.dataclass
class ProbeParams:
    """This is not in use for now."""
    # Zoneplate details - this is not in use
    zp_diam: float = 350
    outerzone: float = 0.05
    beamstop_diam: float = 100  # beam stop diam is nominally 50, but the donut hole looks too small with 50
    cutrad: float = 0.1  # what is cutrad?


@dt.dataclass
class SampleParams:
    """
    The `npix_delta_x` and `npix_delta_y` parameters do not affect the simulation.
    These are only relevant for the reconstruction,
    where this padding reflects the degree of uncertainty about the sample dimensions, and therefore
    the numerical window for the reconstruction.
    For convenience, however, I am storing these within the simulation class so that the reconstruction can easily
    access these values.
    """
    sample_pix_size: float
    wavelength: float
    npix_xy: int = 200
    npix_depth: int = 100
    grain_width: float = 0.5  # in microns, edge-to-edge distance in x (as mounted at HXN)
    grain_height: float = 0.5  # edge-to-edge distance in y
    film_thickness: float = 0.1  # in microns
    npix_delta_x: int = 5  # this padding reflects the uncertainty about either size of the object.
    npix_delta_y: int = 5
    lattice: List[float] = dt.field(default_factory=lambda: np.array([0.0003905] * 3))
    strain_type: str = 'point_inclusion'
    random_scaled_magnitudes: bool = True
    magnitudes_max: float = 1.0
    grain_width_delta: float = dt.field(init=False)
    grain_height_delta: float = dt.field(init=False)
    npix_grain_width_only: int = dt.field(init=False)
    npix_grain_height_only: int = dt.field(init=False)
    #film_thickness_delta: float = dt.field(init=False)


    def __post_init__(self):
        self.npix_grain_height_only = np.ceil(self.grain_height / self.sample_pix_size).astype('int32')
        self.npix_grain_width_only = np.ceil(self.grain_width / self.sample_pix_size).astype('int32')
        self.grain_width_delta = self.grain_width + 2 * self.npix_delta_x * self.sample_pix_size
        self.grain_height_delta = self.grain_height + 2 * self.npix_delta_y * self.sample_pix_size
        #self.film_thickness_delta = self.film_thickness + 2 * self.npix_delta_z * self.sample_pix_size

        if self.strain_type != 'point_inclusion':
            raise NotImplementedError("The code for the other strain types have not been maintained. " \
                                      "Modify and use at your own risk.")
        


@dt.dataclass
class PointInclusionStrainParams:
    # multiple point inclusions in sample.
    # result in a radially exponential strain emanating from inclusion location.
    n_inclusion: int = 1  # Number of inclusions
    inclusion_radius: float = 3.0  # pixels
    alpha: float = 0.025  # radial decay constant of distortion in um
    mag: float = 2e-4  # maximum distortion
    name: str = 'strain3'
    coords_inclusion: list = dt.field(default_factory=list, init=False)


@dt.dataclass
class TwoEdgeSlipSystemStrainParams:
    lattice_constant: float = 1.0
    poisson_ratio: float = 0.29
    ratio: float = 0.25
    name: str = "two_edge_slip"


@dt.dataclass
class PartialEdgeDislocationStrainParams:
    # partial edge dislocation, randomly located terminus, random direction in plane.
    mag: float = 1e-3  # magnitude of distortion
    coord_terminus: list = dt.field(default_factory=list, init=False)
    orientation: float = dt.field(init=False)


@dt.dataclass
class DisplacementFieldWaveStrainParams:
    wavelength: float = 0.05  # displacement field wavelength in microns
    mag: float = 1e-3  # magnitude of the wave
    orientation: float = dt.field(init=False)  # random in plane orientation


@dt.dataclass
class StephanStrainParams:
    mag: float = 1e-3  # scaling factor for strain
