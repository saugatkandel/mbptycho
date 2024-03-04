import numpy as np
from typing import List, Tuple
from scipy.interpolate import RegularGridInterpolator
import joblib
import os

from mbptycho.code.probe import Probes
from mbptycho.code.sample import Sample
from mbptycho.code.params import (
    SimulationParams,
    SampleParams
)


def reloadSimulation(
    fname,
    reload_sim=True,
    reload_sample_only_filename: str = None,
    save_sample_only_filename: str = None,
    new_sim_params: dict = None,
    new_extra_sample_params: dict = None,
):
    if reload_sim:
        if os.path.exists(fname):
            print("File exists. Reloading...")
            sim = joblib.load(fname)
            return sim
        else:
            print("File does not exist. Creating new simulation...")
    print("Creating new simulation...")
    if reload_sample_only_filename is not None:
        sim = Simulation(
            sim_params=new_sim_params,
            extra_sample_params=new_extra_sample_params,
            reload_sample_file=reload_sample_only_filename,
            save_sample_file=save_sample_only_filename,
        )
    elif save_sample_only_filename is not None:
        sim = Simulation(
            sim_params=new_sim_params,
            extra_sample_params=new_extra_sample_params,
            save_sample_file=save_sample_only_filename,
        )
    else:
        sim = Simulation(new_sim_params, new_extra_sample_params)
    print(f"Saving new simulation at {fname}...")
    joblib.dump(sim, fname)
    return sim



class SimulationPerBraggPeak:
    def __init__(
        self,
        rho: np.ndarray,
        probe: np.ndarray,
        theta: float,
        two_theta: float,
        gamma: float,
        sample_params: SampleParams,
        ptycho_scan_positions: List,
        npix_det: int,
        x_full: np.ndarray,
        y_full: np.ndarray,
        z_full: np.ndarray,
        poisson_noise: bool,
        poisson_noise_level: float = 0,
    ):
        self.rho = rho
        self.probe = probe

        self.theta = theta
        self.two_theta = two_theta
        self.gamma = gamma
        self.sample_params = sample_params
        self.ptycho_scan_positions = ptycho_scan_positions
        self.npix_det = npix_det

        self.rotate_theta, self.rotate_two_theta, self.rotate_gamma = self.getRotations(
            theta, two_theta, gamma
        )

        self.ki = np.array([0, 0, 1])[:, None]
        self.ki_rotated = self.rotate_gamma @ self.rotate_two_theta @ self.ki

        self.kf = np.array([0, 0, 1])[:, None]
        self.q = self.kf - self.ki_rotated

        # self.probe = self.truncateProbeZeros(self.probe_untrunc)

        # [self.y_nw, self.x_nw, self.z_nw] = [y_nw, x_nw, z_nw]
        [self.y_full, self.x_full, self.z_full] = [y_full, x_full, z_full]

        # self.YY_nw, self.XX_nw, self.ZZ_nw = np.meshgrid(self.y_nw, self.x_nw, self.z_nw, indexing='ij')
        self.YY_full, self.XX_full, self.ZZ_full = np.meshgrid(
            self.y_full, self.x_full, self.z_full, indexing="ij"
        )
        self.nw_coords_stacked = np.stack(
            (self.YY_full.flatten(), self.XX_full.flatten(), self.ZZ_full.flatten()),
            axis=0,
        )

        # [
        #    self.y_nw, self.x_nw, self.z_nw,
        #    self.YY_nw, self.XX_nw, self.ZZ_nw,
        #    self.nw_coords_stacked
        # ] = self.getNumericalWindowCoordinates(self.rho, self.sample_params.sample_pix_size)

        self.nw_rotated_coords_stacked = (
            self.rotate_gamma
            @ self.rotate_two_theta
            @ self.rotate_theta
            @ self.nw_coords_stacked
        )

        # Identify the rotated coordinates which lie outside the numerical window, and where the field can
        # be set to zero.
        # If we have prior knowledge about the sample size, then we can use the sample dimensions instead of
        # the numerical window dimensions to set the field to zero. This greatly reduces the computational cost.

        self.nw_rotation_mask = self.getNonzeroMaskAfterRotation(
            self.nw_rotated_coords_stacked,
            self.sample_params.grain_height,
            self.sample_params.grain_width,
            self.sample_params.film_thickness,
            self.sample_params.sample_pix_size,
        )
        self.nw_rotated_masked_coords = self.nw_rotated_coords_stacked.T[
            self.nw_rotation_mask
        ].T
        self.nw_rotation_mask_indices = np.where(self.nw_rotation_mask)[0]

        self.nw_rotation_mask_delta = self.getNonzeroMaskAfterRotation(
            self.nw_rotated_coords_stacked,
            self.sample_params.grain_height_delta,
            self.sample_params.grain_width_delta,
            self.sample_params.film_thickness,
            self.sample_params.sample_pix_size,
        )
        self.nw_rotated_masked_delta_coords = self.nw_rotated_coords_stacked.T[
            self.nw_rotation_mask_delta
        ].T
        self.nw_rotation_mask_delta_indices = np.where(self.nw_rotation_mask_delta)[0]

        # Saving all of this information for use in the recons routine.
        [
            self.diffraction_patterns,
            # self.rho_slices,
            # self.probe_slices,
            self.projection_all,
        ] = self.getDiffractionPatternsProbeRoll(
            poisson_noise=poisson_noise, poisson_noise_level=poisson_noise_level
        )


    @staticmethod
    def getRotations(
        theta: float, two_theta: float, gamma: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get rotation matrices corresponding to the provided motor positions.

        The angles provided (and the rotations calculated) do not use the standard clockwise/counterclockwise
        convention. I adapted Stephan's code as is, but I need to check with him to make sure what I am doing here
        actually makes sense. Also need to document these rotations properly.

        Parameters
        ----------
        theta : float
            Theta motor (in radians) (which corresponds to a rotation about the y-axis).
        two_theta : float
            Two-theta motor (in radians)
        gamma : float
            Gamma motor (in radians)
        Returns
        -------
        rotate_theta : array(float)
            3x3 Matrix that gives the rotation for the theta angle (about the y-axis). See `Notes`.
        rotate_two_theta : array(float)
            3x3 rotation matrix for two_theta
        rotate_gamma : array(float)
            3x3 rotation matrix for gamma
        Notes
        -----
        Assumes that the coordinates are stored in (y,x,z) format rather than the typical (x,y,z) format.
        """

        # this is rotation about the y-axis
        rotate_theta = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

        rotate_two_theta = np.array(
            [
                [1, 0, 0],
                [0, np.cos(-two_theta), np.sin(-two_theta)],
                [0, -np.sin(-two_theta), np.cos(-two_theta)],
            ]
        )

        rotate_gamma = np.array(
            [
                [np.cos(-gamma), 0, -np.sin(-gamma)],
                [0, 1, 0],
                [np.sin(-gamma), 0, np.cos(-gamma)],
            ]
        )
        return rotate_theta, rotate_two_theta, rotate_gamma

    # @staticmethod
    # def getNumericalWindowCoordinates(rho: np.ndarray, pix_size: float) -> Tuple[np.ndarray, ...]:
    #    """Calculate the coordinates for the truncated numerical window."""
    #    nyr, nxr, nzr = rho.shape
    #    x_nw = np.arange(-nxr // 2, nxr // 2) * pix_size
    #    y_nw = np.arange(-nyr // 2, nyr // 2) * pix_size
    #    z_nw = np.arange(-nzr // 2, nzr // 2) * pix_size
    #
    #    YY_nw, XX_nw, ZZ_nw = np.meshgrid(y_nw, x_nw, z_nw, indexing='ij')
    #    nw_coords_stacked = np.stack((YY_nw.flatten(), XX_nw.flatten(), ZZ_nw.flatten()), axis=0)
    #    return y_nw, x_nw, z_nw, YY_nw, XX_nw, ZZ_nw, nw_coords_stacked

    @staticmethod
    def getNonzeroMaskAfterRotation(
        nw_rotated_coords: np.ndarray,
        y_size: float,
        x_size: float,
        z_size: float,
        pix_size: float,
    ) -> np.ndarray:
        """After rotating the numerical window, calculate which rotated coordinates lie complelety outside
        the un-rotated numerical window. At these points, the wavefield can be assigned to zero at all ptychographic
        scan positions, even prior to any interpolation.

        Notes
        -----
        The rotated points just outside (within a pixel) the original numerical window are also assumed to have
        non-zero fields. This is so that the interpolation routine can assign a fractional weight based on the
        adjacent unrotated pixel.
        """
        rotated_mask = (
            (np.abs(nw_rotated_coords[0]) < (y_size / 2 + pix_size))
            * (np.abs(nw_rotated_coords[1]) < (x_size / 2 + pix_size))
            * (np.abs(nw_rotated_coords[2]) < (z_size / 2 + pix_size))
        )
        return rotated_mask
        # coords_rotated_masked = coords_rotated.T[rotated_mask].T

    def getDiffractionPatternsProbeRoll(
        self, poisson_noise: bool = False, poisson_noise_level: float = 0
    ):
        """Get simulated diffraction patterns for each scan position by rolling the probe. This should be the
        correct procedure.

        Parameters
        ----------
        poisson_noise_level : float
            Not implemented yet.
        Returns
        -------
        diffraction_patterns : array
            Array of shape (n_scan_positions, det_npix, det_npix)
        Notes
        -----
        This function contains a lot of seemingly incomprehensible slicing and dicing. The goal of all of these is to
        avoid roll operations (which are generally quite expensive).
        """
        if poisson_noise and (poisson_noise_level is not None):
            raise NotImplementedError(
                "Custom level of poisson noise not implemented yet."
            )

        diffraction_patterns = np.zeros(
            (self.ptycho_scan_positions.shape[0], self.npix_det, self.npix_det)
        )

        self.probe_pads = []
        self.probe_slices = []

        fields_before_rotation = []
        projections_all = []
        projection_slices_all = []

        rs = self.rho.shape
        # projection slice based on number of pixels in the detector
        npix_diffy = (rs[0] - self.npix_det) // 2
        npix_diffx = (rs[1] - self.npix_det) // 2
        proj_slice = np.s_[
            npix_diffy : rs[0] - npix_diffy, npix_diffx : rs[1] - npix_diffx
        ]

        for indx, (pcy, pcx) in enumerate(self.ptycho_scan_positions):
            if pcy >= 0:
                slicey = np.s_[0 : rs[0]]
                pady0 = pcy
                pady1 = 0
            else:
                slicey = np.s_[-pcy : rs[0] - pcy]
                pady0 = 0
                pady1 = -pcy

            if pcx >= 0:
                slicex = np.s_[0 : rs[1]]
                padx0 = pcx
                padx1 = 0
            else:
                slicex = np.s_[-pcx : rs[1] - pcx]
                padx0 = 0
                padx1 = -pcx
            slice_this = np.s_[slicey, slicex]

            # This is rolling the probe and padding the ends with zeros. See:
            # https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding
            # probe_this = np.pad(self.probe_full, [[pady0, pady1], [padx0, padx1], [0, 0]], mode='constant')[slice_this]
            probe_this = np.pad(
                self.probe, [[pady0, pady1], [padx0, padx1], [0, 0]], mode="constant"
            )[slice_this]
            self.probe_pads.append([[pady0, pady1], [padx0, padx1], [0, 0]])
            self.probe_slices.append(slice_this)

            # field = self.rho_full * probe_this
            field = self.rho * probe_this

            fields_before_rotation.append(field)
            field_rotated = np.zeros_like(field)

            rgi = RegularGridInterpolator(
                (self.y_full, self.x_full, self.z_full),
                field,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )

            # There is some more efficiency to be gained by only interpolating the scan coordinates between
            # cy0:cy1 and cx0:cx1.
            # We are ignoring that for now.
            field_rotated_masked = rgi(self.nw_rotated_masked_coords.T)

            field_rotated.reshape(-1)[self.nw_rotation_mask] = field_rotated_masked

            projection = field_rotated.sum(axis=2)
            projections_all.append(projection)

            # Need to make sure that the numerical window that corresponds to the detector always contains the
            # volume of interest.
            projection = projection[
                proj_slice
            ]  # [npix_diffy: rs[0] - npix_diffy, npix_diffx: rs[1] - npix_diffx]
            projection_slices_all.append(projection)
            ft = np.fft.fft2(np.fft.fftshift(projection), norm="ortho")
            diffraction_patterns[indx] = np.abs(ft) ** 2
        # diffraction_patterns = np.array(diffraction_patterns)

        if poisson_noise:
            print("Adding poisson noise...")
            diffraction_patterns = np.random.poisson(diffraction_patterns)
        # self.fields_before_rotation = np.array(fields_before_rotation, 'complex64')
        self.projection_slices_all = np.array(projection_slices_all, "complex64")
        self.fields_before_rotation = np.array(fields_before_rotation)
        return diffraction_patterns, projections_all  # np.array(fields_before_rotation)


class Simulation:
    def __init__(
        self,
        sim_params: dict = None,
        reload_sample_file: str = None,
        save_sample_file: str = None,
        extra_sample_params: dict = None,
    ):
        if sim_params is None:
            sim_params = {}
        self.params = SimulationParams(**sim_params)

        if reload_sample_file is not None:
            print(f"Reloading sample from provided file... {reload_sample_file}")
            if os.path.exists(reload_sample_file):
                sample = joblib.load(reload_sample_file)
                print("Sample reloaded.")
                if sample.params.wavelength != self.params.wavelength:
                    print(
                        "current simulation wavelength parameter does not match with reloaded sample."
                    )
                elif sample.params.sample_pix_size != self.params.sample_pix_size:
                    print(
                        "current simulation sample pix size parameter does not match with reloaded sample."
                    )
                else:
                    self.sample = sample
            else:
                print("Supplied sample file does not exist.")

        if not hasattr(self, "sample"):
            print("Creating new sample...")
            self.sample = Sample(
                wavelength=self.params.wavelength,
                sample_pix_size=self.params.sample_pix_size,
                sample_params=extra_sample_params,
            )

            if save_sample_file is not None:
                print(f"Saving sample file.. {save_sample_file}")
                joblib.dump(self.sample, save_sample_file)

        self.rhos = self.sample.setSampleRhos(
            HKL_list=self.params.HKL_list,
            magnitudes_scaling_per_peak=self.params.magnitudes_scaling_per_peak,
            random_scaled_magnitudes=self.params.random_scaled_magnitudes,
            magnitudes_max=self.params.magnitudes_max,
        )

        self.probes_obj = Probes(
            self.params.HKL_list,
            self.params.probes_matlab_h5_file,
            self.params.probe_abs_max,
        )
        # self.probes = [p[:, :, self.sample.nw_zmin:self.sample.nw_zmax] for p in self.probes_obj.probes]
        self.probes = [p for p in self.probes_obj.probes]
        # theta, two_theta, gamma
        motor_list = np.array([-self.HKLToTheta(*HKL) for HKL in self.params.HKL_list])
        # self.motor_list_deg = motor_list
        # self.motor_list_rad = motor_list * np.pi / 180

        # Using these redundant representations just for convenience
        self.thetas_deg, self.two_thetas_deg, self.gammas_deg = (
            motor_list.T * 180 / np.pi
        )
        self.thetas_rad, self.two_thetas_rad, self.gammas_rad = motor_list.T

        self.ptycho_scan_positions = self.getPtychoScanPositions(
            self.params.n_scan_positions, self.params.npix_scan_shift
        )
        self.simulations_per_peak = [
            SimulationPerBraggPeak(
                rho=self.rhos[i],
                # rho_full=self.rhos_full[i],
                probe=self.probes[i],
                # probe_full=self.probes[i],
                theta=self.thetas_rad[i],
                two_theta=self.two_thetas_rad[i],
                gamma=self.gammas_rad[i],
                sample_params=self.sample.params,
                ptycho_scan_positions=self.ptycho_scan_positions,
                npix_det=self.params.npix_det,
                # x_nw=self.sample.x_trunc,
                # y_nw=self.sample.y_trunc,
                # z_nw=self.sample.z_trunc,
                x_full=self.sample.x_full,
                y_full=self.sample.y_full,
                z_full=self.sample.z_full,
                poisson_noise=self.params.poisson_noise,
                poisson_noise_level=self.params.poisson_noise_level,
            )
            for i in range(len(self.rhos))
        ]

    @staticmethod
    def getPtychoScanPositions(n_scan_positions: int, npix_scan_shift: int):
        """Assuming that the scan is along the xy direction, and uses a constant scan step in each direction.

        Working in terms of integer pixel shifts instead of possibly subpixel coordinate shifts.

        Parameters
        ----------
        n_scan_positions: int
            Number of scan positions
        npix_scan_shift : int
            Number of pixels for each scan step.
        Returns
        -------
        scan_grid : ndarray(int)
            List containing the scan positions
        """

        # This somewhat peculiar method to get the grid positions works correctly for both even and odd numbers of
        # scan positions
        nhalf = n_scan_positions / 2
        y = (
            np.ceil(np.arange(-nhalf, nhalf)).astype("int") * npix_scan_shift
        )  # * sample_pix_size
        x = (
            np.ceil(np.arange(-nhalf, nhalf)).astype("int") * npix_scan_shift
        )  # * sample_pix_size
        Yscan, Xscan = np.meshgrid(y, x, indexing="ij")
        scan_grid = np.stack((Yscan, Xscan), axis=-1).reshape(-1, 2)
        return scan_grid

    def HKLToTheta(self, H: int, K: int, L: int):
        """Adapt input for Laue geometry at HXN from APS nanoprobe mounting convention.

        Output is in theta, tth, gam APS NP convention: tth in-plane, gam oop

        Parameters
        ----------
        H : int
        K : int
        L : int
        Returns
        -------
        theta : float
            Theta angle in radians
        two_theta : float
            Two-theta in radians
        gamma : float
            Gamma in radians
        """
        Htemp = K
        Ktemp = L
        Ltemp = H

        H = Htemp
        L = Ltemp
        K = Ktemp

        ang_y = np.arctan2(L, H)
        # what is ang_sym?
        ang_sym = np.arcsin(
            (np.sqrt(H**2 + K**2 + L**2) / 2)
            / (self.sample.params.lattice[0] / self.params.wavelength)
        )
        ang2_sym = 2 * ang_sym

        det_phi = np.arcsin(np.cos(ang_y) / np.sin(np.pi / 2 - ang_sym))
        temp = (np.sin(ang2_sym) * np.cos(det_phi)) ** 2 + np.cos(ang2_sym) ** 2

        theta = np.arcsin(np.cos(np.pi / 2 - ang_sym) / np.sin(ang_y))
        two_theta = np.arcsin(np.sin(ang2_sym) * np.cos(det_phi) / temp**0.5)
        gamma = np.arctan(np.sin(ang2_sym) * np.sin(det_phi) / temp**0.5)

        return np.array([theta, two_theta, gamma])


    @staticmethod
    def truncateProbeZeros(probe: np.ndarray) -> np.ndarray:
        """Truncate the probe beam in the x and y directions by removing xz and yz slices that only contain zeros.

        Add padding of 2 pixels  on each side just because.

        Parameters
        ----------
        probe : ndarray(complex)
            Probe array to truncate
        Returns
        -------
        probe_trunc : ndarray(complex)
            Truncated probe array
        """
        raise NotImplementedError("not in use")
        probe_trunc = probe.copy()
        # the if conditions are just for safety
        probe_new = probe_trunc[~np.all(probe_trunc == 0, axis=(1, 2))]
        if (probe_new.shape[0] + 4) < probe_trunc.shape[0]:
            probe_trunc = np.pad(
                probe_new,
                [[2, 2 + probe_new.shape[0] % 2], [0, 0], [0, 0]],
                mode="constant",
            )

        probe_new = probe_trunc[:, ~np.all(probe_trunc == 0, axis=(0, 2))]
        if (probe_new.shape[1] + 4) < probe_trunc.shape[1]:
            probe_trunc = np.pad(
                probe_new,
                [[0, 0], [2, 2 + probe_new.shape[1] % 2], [0, 0]],
                mode="constant",
            )

        print("Truncated probe shape", probe_trunc.shape)
        return probe_trunc