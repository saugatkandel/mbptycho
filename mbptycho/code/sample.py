import numpy as np
from typing import List

from mbptycho.code.params import (
    SampleParams,
    PointInclusionStrainParams,
    DisplacementFieldWaveStrainParams,
    TwoEdgeSlipSystemStrainParams,
    PartialEdgeDislocationStrainParams
)


class Sample:
    def __init__(
        self, wavelength: float, sample_pix_size: float, sample_params: dict = None
    ):
        if sample_params is None:
            sample_params = {}
        self.params = SampleParams(
            sample_pix_size=sample_pix_size, wavelength=wavelength, **sample_params
        )

        print(
            "Depth of numerical window covers",
            self.params.sample_pix_size * self.params.npix_depth,
            "microns",
        )

        self.x_full = (
            np.arange(-self.params.npix_xy // 2, self.params.npix_xy // 2)
            * self.params.sample_pix_size
        )
        self.y_full = (
            np.arange(-self.params.npix_xy // 2, self.params.npix_xy // 2)
            * self.params.sample_pix_size
        )
        self.z_full = (
            np.arange(-self.params.npix_depth // 2, self.params.npix_depth // 2)
            * self.params.sample_pix_size
        )

        self.YY_full, self.XX_full, self.ZZ_full = np.meshgrid(
            self.y_full, self.x_full, self.z_full, indexing="ij"
        )

        # deltay = self.params.sample_pix_size * self.params.npix_delta_y
        # deltax = self.params.sample_pix_size * self.params.npix_delta_x
        # deltaz = self.params.sample_pix_size * self.params.npix_uncertainty_z
        # z_pad = self.params.sample_pix_size * self.params.npix_pad_z
        # print('uncertainty', deltay, deltax, deltaz)

        self.obj_mask_w_delta = (
            (np.abs(self.YY_full) < (self.params.grain_height_delta / 2))
            * (np.abs(self.XX_full) < (self.params.grain_width_delta / 2))
            * (np.abs(self.ZZ_full) < (self.params.film_thickness / 2))
        )
        self.obj_mask_full = (
            (np.abs(self.YY_full) < (self.params.grain_height / 2))
            * (np.abs(self.XX_full) < (self.params.grain_width / 2))
            * (np.abs(self.ZZ_full) < (self.params.film_thickness / 2))
        )

        self.makeStrains(self.params.strain_type)
        # self.truncateNumericalWindow(pad=2, xy_scan=True)
        self.rhos = None

    def makeStrains(self, strain_type: str = "3"):
        if strain_type == "stephan":
            self._setStephanStrain()
        elif strain_type == "3":
            self._setStrain3()
        elif strain_type == "2":
            self._setStrain2()
        elif strain_type == "point_inclusion":
            self._setStrainPointInclusion()
        elif strain_type == "two_edge_slip":
            self._setStrainTwoEdgeSlipSystem()
        else:
            raise ValueError()

    def _setStrainPointInclusion(self):
        self.strain_params = PointInclusionStrainParams()

        # getting the random terminii
        coords_all = np.stack([self.YY_full, self.XX_full], axis=-1).reshape(-1, 2)

        rp = np.random.randint(
            low=0, high=self.obj_mask_full.sum(), size=self.strain_params.n_inclusion
        )
        terminii = coords_all[self.obj_mask_full.flat, :][rp, :]
        self.strain_params.coords_inclusion = terminii

        self.Ux_full = np.zeros_like(self.XX_full)
        self.Uy_full = np.zeros_like(self.XX_full)
        self.Uz_full = np.zeros_like(self.XX_full)
        for terminus in terminii:
            coords_rel = coords_all - terminus
            r = np.reshape(np.sqrt(np.sum(coords_rel**2, axis=1)), self.XX_full.shape)
            th = np.reshape(
                np.arctan2(coords_rel[:, 0], coords_rel[:, 1]), self.XX_full.shape
            )
            U = self.strain_params.mag * np.exp(-self.strain_params.alpha * r)
            Ux_this = U * np.cos(th)
            Uy_this = U * np.sin(th)
            self.Ux_full += Ux_this
            self.Uy_full += Uy_this

    def _setStrainTwoEdgeSlipSystem(self):
        self.strain_params = TwoEdgeSlipSystemStrainParams()

        coords_all = np.stack([self.YY_full, self.XX_full], axis=-1).reshape(-1, 2)
        coords = coords_all[self.obj_mask_full.flat, :]

        mn = np.min(np.max(coords_all, axis=0))
        theta = np.pi * (-1 + 2 * np.random.rand())

        endpoint = self.strain_params.ratio * mn * np.array([0, -1])

        u1 = self._rotatedEdgeDislocation(
            coords_all,
            endpoint,
            theta,
            self.strain_params.lattice_constant * self.params.lattice[0],
            self.strain_params.poisson_ratio,
        )

        u2 = -self._rotatedEdgeDislocation(
            -coords_all,
            endpoint,
            theta,
            self.strain_params.lattice_constant * self.params.lattice[0],
            self.strain_params.poisson_ratio,
        )
        u_full = u1 + u2
        self.Uy_full = self.obj_mask_full * (u_full[:, 0]).reshape(self.XX_full.shape)
        self.Ux_full = self.obj_mask_full * (u_full[:, 1]).reshape(self.XX_full.shape)
        self.Uz_full = np.zeros_like(self.Ux_full)
        return

    def _rotatedEdgeDislocation(
        self,
        coords_all: np.array,
        endpoint: np.array,
        th: float,
        lattice_const: float,
        poisson_ratio: float,
    ):
        # Flipping the axes from the matlab code.
        rot1 = np.array(
            [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]
        )  # rotation theta
        rot2 = np.array(
            [[-np.sin(th), np.cos(th)], [-np.cos(th), -np.sin(th)]]
        )  # rotation (theta + 90)

        # rotate grid passively
        coords_rot = (rot2 @ (coords_all.T - (rot1.T @ endpoint[:, None]))).T

        rads = np.sqrt(np.sum(coords_rot**2, axis=1))
        th2 = np.arctan2(coords_rot[:, 0], coords_rot[:, 1])

        ux = (lattice_const / 2 / np.pi) * (
            th2 + np.sin(2 * th2) / 4 / (1 - poisson_ratio)
        )
        uy = (lattice_const / 2 / np.pi) * (
            -(1 - 2 * poisson_ratio)
            * 2
            * np.log(rads / lattice_const)
            / (4 * (1 - poisson_ratio))
            + (np.cos(2 * th2) - 1) / (4 * (1 - poisson_ratio))
        )
        u_trans = np.array((uy, ux))

        # rotating back
        u = rot1 @ u_trans
        return u.T  # the transpose is redundant, but I am using it for consistentcy with the other notation.

    def _setStrain3(self):
        self.strain_params = PartialEdgeDislocationStrainParams()
        mag = self.strain_params.mag  # magnintude of distortion

        # getting a random terminus
        coords_all = np.stack([self.YY_full, self.XX_full], axis=-1).reshape(-1, 2)

        rp = np.random.randint(0, high=self.obj_mask_full.sum())

        terminus = coords_all[self.obj_mask_full.flat, :][rp, :]
        self.strain_params.coord_terminus = terminus

        # print(rp, terminus)

        # random in plane orientation for cdw
        theta = np.pi * (-1 + 2 * np.random.random())
        self.strain_params.orientation = theta

        # along edge dislocation
        unit = np.array([np.sin(theta), np.cos(theta)])

        # normal to edge dislocation
        unit_p = np.array([np.sin(theta + np.pi / 2), np.cos(theta + np.pi / 2)])

        coords_rel = coords_all - terminus

        phi = np.arctan2(coords_rel @ unit_p, coords_rel @ unit)
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi

        # print(phi.max(), phi.min())

        mask1 = phi < np.pi / 2
        mask2 = (phi > np.pi / 2) * (phi < 3 * np.pi / 2)
        mask3 = phi > 3 * np.pi / 2

        ux = (
            mask1 * np.cos(theta + np.pi / 2)
            + mask2 * np.cos(theta + phi)
            + mask3 * np.cos(theta + 3 * np.pi / 2)
        )
        uy = (
            mask1 * np.sin(theta + np.pi / 2)
            + mask2 * np.sin(theta + phi)
            + mask3 * np.sin(theta + 3 * np.pi / 2)
        )

        self.Ux_full = self.obj_mask_full * mag * np.reshape(ux, self.XX_full.shape)
        self.Uy_full = self.obj_mask_full * mag * np.reshape(uy, self.YY_full.shape)
        self.Uz_full = (
            np.zeros_like(self.ZZ_full) * self.obj_mask_full * np.sign(self.ZZ_full)
        )

    def _setStrain2(self):
        self.strain_params = DisplacementFieldWaveStrainParams()

        # Randomly oriented displacement field wave in plane
        mag = self.strain_params.mag  # magnintude of distortion
        wavelength = self.strain_params.wavelength  # wavelength in microns

        # random in plane orientation for cdw
        theta = np.pi * (-1 + 2 * np.random.random())
        self.strain_params.orientation = theta

        # random unit vector
        unit = np.array([np.sin(theta), np.cos(theta)])

        xm = self.XX_full.mean()
        xrel = self.XX_full - xm
        ym = self.YY_full.mean()
        yrel = self.YY_full - ym
        pts = np.stack((yrel.flat, xrel.flat), axis=0)

        projection = (unit[None, :] @ pts).T

        U = np.reshape(
            mag * np.cos(np.pi * projection / self.strain_params.wavelength),
            self.XX_full.shape,
        )
        self.Ux_full = U * np.cos(theta) * self.obj_mask_full
        self.Uy_full = U * np.sin(theta) * self.obj_mask_full
        self.Uz_full = (
            np.zeros_like(self.ZZ_full) * self.obj_mask_full * np.sign(self.ZZ_full)
        )

    def _setStephanStrain(self):
        # local lattice displacement in units of microns
        self.Ux_full = (
            self.params.strain_scaling_factor
            * (np.cos(self.XX_full) - 1)
            * self.obj_mask_full
            * np.sign(self.XX_full)
        )
        self.Uy_full = (
            self.params.strain_scaling_factor
            * (np.cos(self.YY_full * 0.5) - 1)
            * self.obj_mask_full
            * np.sign(self.YY_full)
        )
        self.Uz_full = (
            np.zeros_like(self.ZZ_full) * self.obj_mask_full * np.sign(self.ZZ_full)
        )

    def setSampleRhos(
        self,
        HKL_list: List[int] = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 0]]),
        magnitudes_scaling_per_peak: List[float] = None,
        random_scaled_magnitudes: bool = True,
        magnitudes_max: float = 1.0,
    ):
        """Calculate the magnitude and phase profile using the provided magnitudes and HKL list.

        Sets the rhos for the sample class, and also returns the rhos.

        Parameters
        ----------
        HKL_list: array_like(int)
            Should be in the format [[H1, K1, L1], [H2, K2, L2]] where [H1, K1, L1] is a bragg peak.
        magnitude_const: float
            Assuming a constant magnitude throughout the sample, for all incoming wavelengths.
        Returns
        -------
        rhos: array(float)
            Containing the rho profile for each of the supplied bragg peaks.
        """

        magnitudes_full = np.ones(self.YY_full.shape)

        if magnitudes_scaling_per_peak is not None:
            print(
                "Magnitude scaling per peak is supplied. Does not apply random scaling."
            )
            scaling_per_peak = np.array(magnitudes_scaling_per_peak)
        elif random_scaled_magnitudes:
            scaling_per_peak = (
                np.random.random_sample(HKL_list.shape[0])
            ) * magnitudes_max
        else:
            scaling_per_peak = np.ones(HKL_list.shape[0]) * magnitudes_max

        # make phases by doing dot products with Q vectors

        displacements_full = np.array(
            [self.Ux_full, self.Uy_full, self.Uz_full]
        ).reshape(3, -1)
        dot_prod_full = HKL_list @ (displacements_full / self.params.lattice[:, None])

        phase_term_full = np.exp(1j * 2 * np.pi * dot_prod_full).reshape(
            -1, *self.Ux_full.shape
        )

        rhos_full = (
            scaling_per_peak[:, None, None, None]
            * magnitudes_full[None, ...]
            * phase_term_full
        )
        self.rhos = rhos_full * self.obj_mask_full
        return self.rhos 
    

    def _truncateNumericalWindow(self, pad=2, xy_scan=True) -> np.ndarray:
        """
        WARNING: The scipy interpolator produces wonky results if the points we want to interpolate at
        lie close to the boundary of the numerical window. So I am just not using this routine any more.

        Once we define the ptychographic scan positions, and if we have prior knowledge about the size of the
        sample, we can estimate the portion of the probe beam that actually interacts with the sample at any given
        scan position. For simplicity, we can take all the scan positions together to calculate the overall numerical
        window  that captures all possible probe-sample interactions. We can then reduce our simulaiton
        box size to include only this numerical window. This can reduce the computational cost of the simulation.

        For when the ptychographic scan raster grid lies completely on the x-y plane, the same z-numerical window
        applies to every scan position. This numerical window is exactly the sample thickness.

        We use a padding of 2 pixels at the top and bottom of the numerical window, again for no particular reason.

        Other considerations might apply for a multi-angle bragg scan.

        Parameters
        ----------
        ux, uy, uz : ndarray(complex)
            Array of displacements to truncate.
        probe : ndarray (complex)
            Probe array to truncate. Both rho and probe should have identical shapes.
        xy_scan : bool
            Whether the ptychographic scan grid lies on the xy plane.
        Returns
        -------
        ux_trunc, uy_trunc, uz_trunc : ndarray(complex)
            Truncated displacement arrays.
        probe_trunc : ndarray(complex)
            Truncated probe array.
        """
        raise NotImplementedError(
            "Scipy interpolation routine is wonky for coordinates near numerical window edges."
        )
        if not xy_scan:
            raise NotImplementedError(
                "only implemented for raster scan along the xy directions"
            )
        # pad = 20
        non_zero_z = np.where(np.sum(self.obj_mask_full, axis=(0, 1)))[0]
        self.nw_zmin = np.maximum(0, non_zero_z.min() - pad)
        self.nw_zmax = np.minimum(self.obj_mask_full.shape[2], non_zero_z.max() + pad)
        # self.Ux_trunc = self.Ux_full[:, :, self.nw_zmin: self.nw_zmax]
        # self.Uy_trunc = self.Uy_full[:, :, self.nw_zmin: self.nw_zmax]
        # self.Uz_trunc = self.Uz_full[:, :, self.nw_zmin: self.nw_zmax]
        # self.obj_mask_trunc = self.obj_mask_full[:, :, self.nw_zmin: self.nw_zmax]
        # self.obj_mask_trunc_no_pad = self.obj_mask_full_no_pad[:, :, self.nw_zmin: self.nw_zmax]

        non_zero_y = np.where(np.sum(self.obj_mask_full, axis=(1, 2)))[0]
        self.nw_ymin = 0
        self.nw_ymax = self.obj_mask_full.shape[0]
        # self.nw_ymin = np.maximum(0, non_zero_y.min() - pad)
        # self.nw_ymax = np.minimum(self.obj_mask_full.shape[0],
        #                          non_zero_y.max() + pad)

        non_zero_y_delta = np.where(np.sum(self.obj_mask_w_delta, axis=(1, 2)))[0]
        # self.nw_ymin_delta = np.maximum(0, non_zero_y_delta.min() - pad)
        # self.nw_ymax_delta = np.minimum(self.obj_mask_w_delta.shape[0],
        #                                non_zero_y_delta.max() + pad)
        self.nw_ymin_delta = 0
        self.nw_ymax_delta = self.obj_mask_w_delta.shape[0]

        non_zero_x = np.where(np.sum(self.obj_mask_full, axis=(0, 2)))[0]
        # self.nw_xmin = np.maximum(0, non_zero_x.min() - pad)
        # self.nw_xmax = np.minimum(self.obj_mask_full.shape[1],
        #                          non_zero_x.max() + pad)
        self.nw_xmin = 0
        self.nw_xmax = self.obj_mask_full.shape[1]

        non_zero_x_delta = np.where(np.sum(self.obj_mask_w_delta, axis=(0, 2)))[0]
        # self.nw_xmin_delta = np.maximum(0, non_zero_x_delta.min() - pad)
        # self.nw_xmax_delta = np.minimum(self.obj_mask_w_delta.shape[1],
        #                                non_zero_x_delta.max() + pad)
        self.nw_xmin_delta = 0
        self.nw_xmax_delta = self.obj_mask_w_delta.shape[1]

        slice_trunc = np.s_[
            self.nw_ymin : self.nw_ymax,
            self.nw_xmin : self.nw_xmax,
            self.nw_zmin : self.nw_zmax,
        ]

        slice_delta = np.s_[
            self.nw_ymin_delta : self.nw_ymax_delta,
            self.nw_xmin_delta : self.nw_xmax_delta,
            self.nw_zmin : self.nw_zmax,
        ]
        print(self.nw_zmax, self.XX_full.shape)
        print(slice_trunc, slice_delta)

        self.Ux_trunc = self.Ux_full[slice_trunc]
        self.Uy_trunc = self.Uy_full[slice_trunc]
        self.Uz_trunc = self.Uz_full[slice_trunc]
        self.obj_mask_trunc_delta = self.obj_mask_w_delta[slice_delta]
        self.obj_mask_trunc = self.obj_mask_full[slice_trunc]

        self.x_trunc = self.x_full[self.nw_xmin : self.nw_xmax]
        self.y_trunc = self.y_full[self.nw_ymin : self.nw_ymax]
        self.z_trunc = self.z_full[self.nw_zmin : self.nw_zmax]

        self.x_trunc_delta = self.x_full[self.nw_xmin_delta : self.nw_xmax_delta]
        self.y_trunc_delta = self.y_full[self.nw_ymin_delta : self.nw_ymax_delta]
        # self.z_trunc_delta = self.z_full[self.nw_zmin: self.nw_zmax]

        self.YY_trunc, self.XX_trunc, self.ZZ_trunc = np.meshgrid(
            self.y_trunc, self.x_trunc, self.z_trunc, indexing="ij"
        )

        # Along the z-direciton only
        # sample_half_width_npix = np.ceil(self.sample_params.film_thickness
        #                                 / (self.sample_params.sample_pix_size * 2)).astype('int')
        # numerical_window_center = rho.shape[-1] // 2

        # the minimum/maximum here is just for added safety
        # miniz = np.maximum(numerical_window_center - sample_half_width_npix - 2, 0)
        # maxiz = np.minimum(numerical_window_center + sample_half_width_npix + 2, rho.shape[-1])
        #
        # probe_trunc = probe[:,:,miniz : maxiz]
        # ux_trunc = rho[:,:,miniz : maxiz]
        # print("Original numerical window", rho.shape)
        # print("Truncated numerical window", rho_trunc.shape)
        # return rho_trunc, probe_trunc

