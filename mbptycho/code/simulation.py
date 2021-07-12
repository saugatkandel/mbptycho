import numpy as np
import dataclasses as dt
from typing import List, Tuple
from scipy.interpolate import RegularGridInterpolator
import dill
import os
import abc
import gzip

@dt.dataclass
class SimulationParams:
    wavelength: float =  1.377e-4 #microns, for 9 keV
    det_pix_size: float = 55 #microns, for Merlin
    det_dist = 0.35e6 # detector to sample distance
    npix_det: int = 150 # square detector
    HKL_list: List[int] = dt.field(default_factory=lambda: np.array([[1, 0, 0],
                                                                     [1, 1, 0],
                                                                     [1, 2, 0]]))
    magnitudes_scaling_per_peak: List[int] = None
    random_scaled_magnitudes: bool = True
    magnitudes_max: float = 0.5
    n_scan_positions: int = 11
    npix_scan_shift: int = 4
    sample_pix_size: float = dt.field(init=False)
    probes_matlab_file: str = '../matlab/Datasets/probes_015.mat'
    poisson_noise: bool = True
    poisson_noise_level: float = None # not implemented yet
    
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
    beamstop_diam: float = 100 # beam stop diam is nominally 50, but the donut hole looks too small with 50
    cutrad: float = 0.1 # what is cutrad?
    

@dt.dataclass
class SampleParams:
    sample_pix_size: float
    wavelength: float
    npix_xy: int = 200
    npix_depth: int = 100 
    grain_width = 0.8 # in microns, edge-to-edge distance in x (as mounted at HXN)
    grain_height = 0.5 # edge-to-edge distance in y
    film_thickness: float = 0.1 # in microns
    npix_pad_x: int = 0
    npix_pad_y: int = 0
    lattice: List[float] = dt.field(default_factory=lambda: np.array([0.0003905] * 3))
    strain_type: str = 'point_inclusion'
    random_scaled_magnitudes: bool = True
    magnitudes_max: float = 1.0


@dt.dataclass
class PointInclusionStrainParams:
    # multiple point inclusions in sample.
    # result in a radially exponential strain emanating from inclusion location.
    n_inclusion: int = 1 # Number of inclusions
    inclusion_radius: float = 3.0 # pixels
    alpha: float = 0.025 # radial decay constant of distortion in um
    mag: float = 2e-4 #maximum distortion
    name: str ='strain3'
    coords_inclusion: list = dt.field(default_factory=list, init=False)   

@dt.dataclass
class PartialEdgeDislocationStrainParams:
    # partial edge dislocation, randomly located terminus, random direction in plane.
    mag: float = 1e-3 # magnitude of distortion
    coord_terminus: list = dt.field(default_factory=list, init=False)
    orientation: float = dt.field(init=False)


@dt.dataclass
class DisplacementFieldWaveStrainParams:
    wavelength: float = 0.05 # displacement field wavelength in microns
    mag: float = 1e-3 # magnitude of the wave
    orientation: float = dt.field(init=False) # random in plane orientation
    

@dt.dataclass
class StephanStrainParams:
    mag: float = 1e-3 # scaling factor for strain 
    

def reloadSimulation(fname, 
                     reload_sim=True,
                     reload_sample_only_filename:str=None,
                     save_sample_only_filename:str=None,
                     new_sim_params:dict = None, 
                     new_extra_sample_params:dict = None):

    if reload_sim:
        if os.path.exists(fname):
            print('File exists. Reloading...')
            with open(fname, 'rb') as f:
                sim = dill.load(f)
            return sim
        else:
            print('File does not exist. Creating new simulation...')
    print('Creating new simulation...')
    if reload_sample_only_filename is not None:
        sim = Simulation(sim_params=new_sim_params, extra_sample_params=new_extra_sample_params, 
                         reload_sample_file=reload_sample_only_filename,
                         save_sample_file=save_sample_only_filename)
    elif save_sample_only_filename is not None:
        sim = Simulation(sim_params=new_sim_params, extra_sample_params=new_extra_sample_params,
                         save_sample_file=save_sample_only_filename)
    else:
        sim = Simulation(new_sim_params, new_extra_sample_params)
    print('Saving new simulation...')
    with open(fname, 'wb') as f:
        dill.dump(sim, f)
    return sim
        

class Sample:
    def __init__(self, wavelength: float, sample_pix_size: float, sample_params: dict=None):

        if sample_params is None:
            sample_params = {}
        self.params = SampleParams(sample_pix_size=sample_pix_size,
                                   wavelength=wavelength,
                                   **sample_params)

        print("Depth of numerical window covers", self.params.sample_pix_size * self.params.npix_depth, "microns")
        
        self.x_full = np.arange(-self.params.npix_xy // 2, self.params.npix_xy // 2) * self.params.sample_pix_size
        self.y_full = np.arange(-self.params.npix_xy // 2, self.params.npix_xy // 2) * self.params.sample_pix_size
        self.z_full = (np.arange(-self.params.npix_depth // 2, self.params.npix_depth // 2) *
                       self.params.sample_pix_size)
        
        self.YY_full, self.XX_full, self.ZZ_full = np.meshgrid(self.y_full, self.x_full, self.z_full, indexing='ij')

        y_pad = self.params.sample_pix_size * self.params.npix_pad_y
        x_pad = self.params.sample_pix_size * self.params.npix_pad_x
        #z_pad = self.params.sample_pix_size * self.params.npix_pad_z
        print('pads', y_pad, x_pad)
        self.obj_mask_full = ((np.abs(self.YY_full) < (self.params.grain_height / 2 + y_pad))
                         * (np.abs(self.XX_full) < (self.params.grain_width / 2 + x_pad))
                         * (np.abs(self.ZZ_full) < (self.params.film_thickness / 2 )))
        self.obj_mask_full_no_pad = ((np.abs(self.YY_full) < (self.params.grain_height / 2))
                                * (np.abs(self.XX_full) < (self.params.grain_width / 2))
                                * (np.abs(self.ZZ_full) < (self.params.film_thickness / 2)))
        
        self.makeStrains(self.params.strain_type)
        self.truncateNumericalWindow(pad=2, xy_scan=True)
        self.rhos = None
        
    
    def makeStrains(self, strain_type:str='3'):
        
        if strain_type=='stephan':
            self._setStephanStrain()
        elif strain_type == '3':
            self._setStrain3()
        elif strain_type == '2':
            self._setStrain2()
        elif strain_type == 'point_inclusion':
            self._setStrainPointInclusion()
        else:
            raise ValueError()
    
    def _setStrainPointInclusion(self):
        self.strain_params = PointInclusionStrainParams()
        
        # getting the random terminii
        coords_all = np.stack([self.YY_full, self.XX_full], axis=-1).reshape(-1, 2)
        
        rp = np.random.randint(low=0, high=self.obj_mask_full_no_pad.sum(),
                               size=self.strain_params.n_inclusion)
        terminii = coords_all[self.obj_mask_full_no_pad.flat, :][rp, :]
        self.strain_params.coords_inclusion = terminii
        
        self.Ux_full = np.zeros_like(self.XX_full)
        self.Uy_full = np.zeros_like(self.XX_full)
        self.Uz_full = np.zeros_like(self.XX_full)
        for terminus in terminii:
            coords_rel = coords_all - terminus
            r = np.reshape(np.sqrt(np.sum(coords_rel**2, axis=1)), self.XX_full.shape)
            th = np.reshape(np.arctan2(coords_rel[:,0], coords_rel[:,1]), self.XX_full.shape)
            U = self.strain_params.mag * np.exp(-self.strain_params.alpha * r)
            Ux_this = U * np.cos(th)
            Uy_this = U * np.sin(th)
            self.Ux_full += Ux_this
            self.Uy_full += Uy_this
            
    
    def _setStrain3(self):
        self.strain_params = PartialEdgeDislocationStrainParams()
        mag = self.strain_params.mag # magnintude of distortion
        
        # getting a random terminus
        coords_all = np.stack([self.YY_full, self.XX_full], axis=-1).reshape(-1, 2)
        
        rp = np.random.randint(0, high=self.obj_mask_full_no_pad.sum())
        
        terminus = coords_all[self.obj_mask_full_no_pad.flat,:][rp,:]
        self.strain_params.coord_terminus = terminus
        
        #print(rp, terminus)
    
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
        
        #print(phi.max(), phi.min())
        
        mask1 = phi < np.pi / 2
        mask2 = (phi > np.pi / 2) * (phi < 3 * np.pi / 2)
        mask3 = phi > 3 * np.pi / 2
        
        ux = (mask1 * np.cos(theta + np.pi / 2) + mask2 * np.cos(theta + phi) 
              + mask3 * np.cos(theta + 3 * np.pi / 2))
        uy = (mask1 * np.sin(theta + np.pi / 2) + mask2 * np.sin(theta + phi) 
              + mask3 * np.sin(theta + 3 * np.pi / 2))
        
        self.Ux_full = self.obj_mask_full_no_pad * mag * np.reshape(ux, self.XX_full.shape)
        self.Uy_full = self.obj_mask_full_no_pad * mag * np.reshape(uy, self.YY_full.shape)
        self.Uz_full = np.zeros_like(self.ZZ_full) * self.obj_mask_full_no_pad * np.sign(self.ZZ_full)
        
    def _setStrain2(self):
        self.strain_params = DisplacementFieldWaveStrainParams()
        
        # Randomly oriented displacement field wave in plane
        mag = self.strain_params.mag # magnintude of distortion
        wavelength = self.strain_params.wavelength #  wavelength in microns
    
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
        
        U = np.reshape(mag * np.cos(np.pi * projection / self.strain_params.wavelength), self.XX_full.shape)
        self.Ux_full = U * np.cos(theta) * self.obj_mask_full_no_pad
        self.Uy_full = U * np.sin(theta) * self.obj_mask_full_no_pad
        self.Uz_full = np.zeros_like(self.ZZ_full) * self.obj_mask_full_no_pad * np.sign(self.ZZ_full)
        
        
    def _setStephanStrain(self):
        
        # local lattice displacement in units of microns
        self.Ux_full = (self.params.strain_scaling_factor * (np.cos(self.XX_full) - 1) *
                        self.obj_mask_full_no_pad * np.sign(self.XX_full))
        self.Uy_full = (self.params.strain_scaling_factor * (np.cos(self.YY_full * 0.5) - 1) *
                        self.obj_mask_full_no_pad * np.sign(self.YY_full))
        self.Uz_full = np.zeros_like(self.ZZ_full) * self.obj_mask_full_no_pad * np.sign(self.ZZ_full)
    
    def setSampleRhos(self,
                      hole_center: List[int] = np.array([0,0,0]),
                      HKL_list: List[int] = np.array([[1, 0, 0],
                                                      [1, 1, 0],
                                                      [1, 2, 0]]),
                      magnitudes_scaling_per_peak: List[float] = None,
                      random_scaled_magnitudes: bool = True,
                      magnitudes_max: float = 1.0):
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
        
        magnitudes = np.ones(self.YY_trunc.shape)

        if self.params.strain_type == '1':# or self.params.strain_type=='point_inclusion':
            for terminus in self.strain_params.coords_inclusion:
                radius_sq = (self.YY_trunc - terminus[0])**2 + (self.XX_trunc - terminus[1])**2
                mask = radius_sq > (self.strain_params.inclusion_radius * self.params.sample_pix_size)**2
                magnitudes *= mask

        if magnitudes_scaling_per_peak is not None:
            print("Magnitude scaling per peak is supplied. Does not apply random scaling.")
            scaling_per_peak = np.array(magnitudes_scaling_per_peak)
        elif random_scaled_magnitudes:
            scaling_per_peak = (np.random.random_sample(HKL_list.shape[0])) * magnitudes_max
        else:
            scaling_per_peak = np.ones(HKL_list.shape[0]) * magnitudes_max

        # make phases by doing dot products with Q vectors
        displacements = np.array([self.Ux_trunc, self.Uy_trunc, self.Uz_trunc]).reshape(3, -1) 
        dot_prod = HKL_list @ (displacements / self.params.lattice[:,None])
        
        phase_term = np.exp(1j * 2 * np.pi * dot_prod).reshape(-1, *self.Ux_trunc.shape)
        rhos = scaling_per_peak[:,None,None,None] * magnitudes[None, ...] * phase_term
        self.rhos = rhos * self.obj_mask_trunc_no_pad
        self.magnitudes_trunc_mask = magnitudes.astype('bool')

        return self.rhos
    
    def truncateNumericalWindow(self, pad=2, xy_scan=True) -> np.ndarray:
        """Once we define the ptychographic scan positions, and if we have prior knowledge about the size of the
        sample, we can estimate the portion of the probe beam that actually interacts with the sample at any given
        scan position. For simplicity, we can take all the scan positions together to calculate the overall numerical 
        window  that captures all possible probe-sample interactions. We can then reduce our simulaiton 
        box size to include only this numerical window. This can reduce the computational cost of the simulation.
        
        For when the ptychographic scan raster grid lies completely on the x-y plane, the same z-numerical window 
        applies to every scan position. This numerical window is exactly the sample thickness. 
        
        For now, we don't truncate the numerical window along the x and y directions (for no particular reason).
        We also use a padding of 2 pixels at the top and bottom, again for no particular reason.
        
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
        if not xy_scan:
            raise NotImplementedError("only implemented for raster scan along the xy directions")
        
        non_zero_z = np.where(np.sum(self.obj_mask_full, axis=(0,1)))[0]
        self.nw_zmin = np.maximum(0, non_zero_z.min() - pad)
        self.nw_zmax = np.minimum(self.obj_mask_full.shape[2], non_zero_z.max() + pad)
        self.Ux_trunc = self.Ux_full[:, :, self.nw_zmin: self.nw_zmax]
        self.Uy_trunc = self.Uy_full[:, :, self.nw_zmin: self.nw_zmax]
        self.Uz_trunc = self.Uz_full[:, :, self.nw_zmin: self.nw_zmax]
        self.obj_mask_trunc = self.obj_mask_full[:, :, self.nw_zmin: self.nw_zmax]
        self.obj_mask_trunc_no_pad = self.obj_mask_full_no_pad[:, :, self.nw_zmin: self.nw_zmax]
        
        self.x_trunc = self.x_full
        self.y_trunc = self.y_full
        self.z_trunc = self.z_full[self.nw_zmin: self.nw_zmax]
        
        self.YY_trunc, self.XX_trunc, self.ZZ_trunc = np.meshgrid(self.y_trunc, self.x_trunc, self.z_trunc,
                                                                  indexing='ij')
        
        # Along the z-direciton only
        #sample_half_width_npix = np.ceil(self.sample_params.film_thickness 
        #                                 / (self.sample_params.sample_pix_size * 2)).astype('int')
        #numerical_window_center = rho.shape[-1] // 2
        
        # the minimum/maximum here is just for added safety
        #miniz = np.maximum(numerical_window_center - sample_half_width_npix - 2, 0)
        #maxiz = np.minimum(numerical_window_center + sample_half_width_npix + 2, rho.shape[-1])
        #
        #probe_trunc = probe[:,:,miniz : maxiz]
        #ux_trunc = rho[:,:,miniz : maxiz]
        #print("Original numerical window", rho.shape)
        #print("Truncated numerical window", rho_trunc.shape)
        #return rho_trunc, probe_trunc
    
class Probes:
    def __init__(self, HKL_list: List, probes_matlab_file: str):
        probes_all = self.loadFullBeamProfiles(probes_matlab_file)
        probes_selected = []
        for (H,K,L) in HKL_list:
            if (H == 1 and K == 0 and L == 0):
                probes_selected.append(probes_all[0])
            elif (H == 1 and K == 1 and L == 0):
                probes_selected.append(probes_all[1])
            elif (H == 1 and K == 2 and L == 0):
                probes_selected.append(probes_all[2])
            elif (H==2 and K==1 and L==0):
                probes_selected.append(probes_all[3])
            elif (H==3 and K==1 and L==0):
                probes_selected.append(probes_all[4])
            else:
                raise NotImplementedError("not implemented")
        self.probes = probes_selected
        
    def loadFullBeamProfiles(self, probes_matlab_file):
        from scipy import io as scio
        print("Loading probe from matlab data...")
        probes_all = scio.loadmat(probes_matlab_file)['probes']
        probes_all = np.array([probes_all[0][i][0] for i in range(probes_all[0].shape[0])]) 
        print("Loading successful...")
        return probes_all
    
    def generateFrom2DProfile(self, pixel_size):
        raise NotImplementedError("Not supported yet")
        
        probe = scio.loadmat('../recon_probe_29902.mat')['prb']
        probe = np.flipud(probe) / np.max(np.abs(probe)) # This gives me a probe structure matching the matlab output
        probe_pixel_size = 10e-3

        x = np.arange(-50, 50) * pixel_size
        Y, X = np.meshgrid(x, x, indexing='ij')

        mask = np.abs(probe) > 0.01#25

        vertsprobe1 = np.stack([Y[mask], X[mask], -2 * np.ones(mask.sum())], axis=1)
        vertsprobe2 = np.stack([Y[mask], X[mask], 2 * np.ones(mask.sum())], axis=1)
        vertsprobe = np.concatenate([vertsprobe1, vertsprobe2], axis=0)
        
class SimulationPerBraggPeak:
    def __init__(self, rho: np.ndarray,
                 probe: np.ndarray,
                 theta: float,
                 two_theta: float,
                 gamma: float,
                 sample_params: SampleParams,
                 ptycho_scan_positions: List,
                 npix_det: int,
                 x_nw: np.ndarray,
                 y_nw: np.ndarray,
                 z_nw: np.ndarray,
                 poisson_noise: bool, 
                 poisson_noise_level: float=0):
        self.rho = rho
        self.probe = probe
        self.theta = theta
        self.two_theta = two_theta
        self.gamma = gamma
        self.sample_params = sample_params
        self.ptycho_scan_positions = ptycho_scan_positions
        self.npix_det = npix_det
        
        self.rotate_theta, self.rotate_two_theta, self.rotate_gamma = self.getRotations(theta, two_theta, gamma) 
        
        self.ki = np.array([0, 0, 1])[:, None]
        self.ki_rotated = self.rotate_gamma @ self.rotate_two_theta @ self.ki
        
        self.kf = np.array([0, 0, 1])[:, None]
        self.q = self.kf - self.ki_rotated
        
        
        self.probe = self.truncateProbeZeros(self.probe)
        
        self.y_nw = y_nw
        self.x_nw = x_nw
        self.z_nw = z_nw
        self.YY_nw, self.XX_nw, self.ZZ_nw = np.meshgrid(self.y_nw, self.x_nw, self.z_nw, indexing='ij')
        self.nw_coords_stacked = np.stack((self.YY_nw.flatten(), self.XX_nw.flatten(), self.ZZ_nw.flatten()), axis=0)
        
        #[
        #    self.y_nw, self.x_nw, self.z_nw,
        #    self.YY_nw, self.XX_nw, self.ZZ_nw,
        #    self.nw_coords_stacked
        #] = self.getNumericalWindowCoordinates(self.rho, self.sample_params.sample_pix_size)
        
        self.nw_rotated_coords_stacked = (self.rotate_gamma @ self.rotate_two_theta @ 
                                          self.rotate_theta @ self.nw_coords_stacked)
        
        # Identify the rotated coordinates which lie outside the numerical window, and where the field can
        # be set to zero.
        # If we have prior knowledge about the sample size, then we can use the sample dimensions instead of 
        # the numerical window dimensions to set the field to zero. This greatly reduces the computational cost.
        
        self.nw_rotation_mask = self.getNonzeroMaskAfterRotation(self.nw_rotated_coords_stacked,
                                                                 self.sample_params.grain_height,
                                                                 self.sample_params.grain_width,
                                                                 self.sample_params.film_thickness,
                                                                 self.sample_params.sample_pix_size)
        self.nw_rotated_masked_coords = self.nw_rotated_coords_stacked.T[self.nw_rotation_mask].T
        self.nw_rotation_mask_indices = np.where(self.nw_rotation_mask)[0]
        
        # Saving all of this information for use in the recons routine.
        [
            self.diffraction_patterns,
            self.rho_slices,
            self.probe_slices,
            self.proj_slices
        ] = self.getDiffractionPatterns(poisson_noise=poisson_noise, poisson_noise_level=poisson_noise_level)
        
    
    
    
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
        
        probe_trunc = probe.copy()
        # the if conditions are just for safety
        probe_new = probe_trunc[~np.all(probe_trunc==0, axis=(1, 2))]
        if (probe_new.shape[0] + 4) < probe_trunc.shape[0]:
            probe_trunc = np.pad(probe_new, [[2, 2 + probe_new.shape[0] % 2],
                                             [0, 0],
                                             [0, 0]], mode="constant")

        
        probe_new = probe_trunc[:, ~np.all(probe_trunc==0, axis=(0, 2))]
        if (probe_new.shape[1] + 4) < probe_trunc.shape[1]:
            probe_trunc = np.pad(probe_new, [[0, 0],
                                             [2, 2 + probe_new.shape[1] % 2],
                                             [0, 0]], mode="constant")
        
        print("Truncated probe shape", probe_trunc.shape)
        return probe_trunc
    
    @staticmethod
    def getRotations(theta: float, two_theta: float, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        rotate_theta =  np.array([[1, 0, 0],
                                  [0, np.cos(theta), np.sin(theta)],
                                  [0, -np.sin(theta), np.cos(theta)]])

        rotate_two_theta = np.array([[1, 0, 0],
                                     [0, np.cos(-two_theta), np.sin(-two_theta)],
                                     [0, -np.sin(-two_theta), np.cos(-two_theta)]])

        rotate_gamma = np.array([[np.cos(-gamma), 0, -np.sin(-gamma)],
                                 [0, 1, 0],
                                 [np.sin(-gamma), 0, np.cos(-gamma)]])
        return rotate_theta, rotate_two_theta, rotate_gamma
    
    #@staticmethod
    #def getNumericalWindowCoordinates(rho: np.ndarray, pix_size: float) -> Tuple[np.ndarray, ...]:
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
    def getNonzeroMaskAfterRotation(nw_rotated_coords: np.ndarray,
                                   y_size: float,
                                   x_size: float,
                                   z_size: float,
                                   pix_size: float) -> np.ndarray:
        """After rotating the numerical window, calculate which rotated coordinates lie complelety outside 
        the un-rotated numerical window. At these points, the wavefield can be assigned to zero at all ptychographic 
        scan positions, even prior to any interpolation. 
        
        Notes
        -----
        The rotated points just outside (within a pixel) the original numerical window are also assumed to have
        non-zero fields. This is so that the interpolation routine can assign a fractional weight based on the 
        adjacent unrotated pixel.
        """
        rotated_mask = ((np.abs(nw_rotated_coords[0]) < (y_size / 2 + pix_size))
                        * (np.abs(nw_rotated_coords[1]) < (x_size / 2 + pix_size))
                        * (np.abs(nw_rotated_coords[2]) < (z_size / 2 + pix_size)))
        return rotated_mask
        #coords_rotated_masked = coords_rotated.T[rotated_mask].T
        
    def getDiffractionPatterns(self, poisson_noise: bool = False, poisson_noise_level: float=0):
        """Get simulated diffraction patterns for each scan position.
        
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
            raise NotImplementedError("Custom level of poisson noise not implemented yet.")
        
        diffraction_patterns = []
        rho_slices = []
        probe_slices = []
        projection_slices = []
        for pcy, pcx in self.ptycho_scan_positions:
            
            # Calculating the scan slice
            cy0 = self.rho.shape[0]// 2 - self.probe.shape[0] // 2 + pcy
            cy1 = cy0 + self.probe.shape[0]#+ self.probe.shape[0] // 2 + pcy

            cx0 = self.rho.shape[1]// 2 - self.probe.shape[1] // 2 + pcx
            cx1 = cx0 + self.probe.shape[1]#self.rho.shape[1]// 2 + pcx + self.probe.shape[1]#self.probe.shape[1] // 2 + pcx
            # Scan center
            centery = (cy0 + cy1) // 2
            centerx = (cx0 + cx1) // 2
            
            field = np.zeros_like(self.rho)
            
            # The next code block ensures we always stay within the numerical window.
            # We do this *after* the projection center is calculated so that
            # the scan center is calculated correctly.
            # There is probably a much easier way to do this...
            rho_slice = np.s_[np.maximum(0, cy0): np.minimum(self.rho.shape[0], cy1),
                              np.maximum(0, cx0): np.minimum(self.rho.shape[1], cx1)]
            
            probe_slice = np.s_[-np.minimum(0, cy0): self.probe.shape[0] + np.minimum(0, self.rho.shape[0] - cy1),
                                -np.minimum(0, cx0): self.probe.shape[1] + np.minimum(0, self.rho.shape[1] - cx1)]
            
            field[rho_slice] = self.probe[probe_slice] * self.rho[rho_slice]
            
            field_rotated = np.zeros_like(field)
            rgi = RegularGridInterpolator((self.y_nw, self.x_nw, self.z_nw),
                                          field,
                                          method='linear',
                                          bounds_error=False,
                                          fill_value=0)
            
            # There is some more efficiency to be gained by only interpolating the scan coordinates between
            # cy0:cy1 and cx0:cx1.
            # We are ignoring that for now.
            field_rotated_masked = rgi(self.nw_rotated_masked_coords.T)

            field_rotated.reshape(-1)[self.nw_rotation_mask] = field_rotated_masked
            
            projection = field_rotated.sum(axis=2)
            
            # Need to make sure that the numerical window that corresponds to the detector always contains the
            # volume of interest.
            
            # projection slice based on number of pixels in the detector
            proj_cy0 = centery - self.npix_det // 2
            proj_cy1 = centery + self.npix_det // 2
            proj_cx0 = centerx - self.npix_det // 2
            proj_cx1 = centerx + self.npix_det // 2
            proj_slice = np.s_[proj_cy0:proj_cy1, proj_cx0:proj_cx1]
            projection = projection[proj_slice]
            #print(centery, centerx, proj_cy0, proj_cy1, proj_cx0, proj_cx1)

            rho_slices.append(rho_slice)
            probe_slices.append(probe_slice)
            projection_slices.append(proj_slice)
            
            ft = np.fft.fft2(projection)
            diffraction_patterns.append(np.abs(ft)**2)
        diffraction_patterns = np.array(diffraction_patterns)
        if poisson_noise:
            print("Adding poisson noise...")
            diffraction_patterns = np.random.poisson(diffraction_patterns)
            
        return diffraction_patterns, rho_slices, probe_slices, projection_slices 
        
class Simulation:
    def __init__(self, sim_params:dict = None, 
                 reload_sample_file:str=None,
                 save_sample_file:str=None,
                 extra_sample_params:dict = None):
        
        if sim_params is None:
            sim_params = {}
        self.params = SimulationParams(**sim_params)
        
        
        if reload_sample_file is not None:
            print(f"Reloading sample from provided file... {reload_sample_file}")
            if os.path.exists(reload_sample_file):
                with open(reload_sample_file, 'rb') as f:
                    sample = dill.load(f)
                print('Sample reloaded.')
                if sample.params.wavelength != self.params.wavelength:
                    print("current simulation wavelength parameter does not match with reloaded sample.")
                elif sample.params.sample_pix_size != self.params.sample_pix_size:
                    print("current simulation sample pix size parameter does not match with reloaded sample.")
                else:
                    self.sample = sample
            else:
                print("Supplied sample file does not exist.")

        if not hasattr(self, "sample"):
            print('Creating new sample...')
            self.sample = Sample(wavelength=self.params.wavelength, sample_pix_size=self.params.sample_pix_size, sample_params=extra_sample_params)

            if save_sample_file is not None:
                print(f"Saving sample file.. {save_sample_file}")
                with open(save_sample_file, 'wb') as f:
                    dill.dump(self.sample, f)
        
        self.rhos = self.sample.setSampleRhos(
            HKL_list=self.params.HKL_list,
            magnitudes_scaling_per_peak=self.params.magnitudes_scaling_per_peak,
            random_scaled_magnitudes=self.params.random_scaled_magnitudes,
            magnitudes_max=self.params.magnitudes_max)
        
        self.probes_obj = Probes(self.params.HKL_list, self.params.probes_matlab_file)
        self.probes = [p[:,:,self.sample.nw_zmin:self.sample.nw_zmax] for p in self.probes_obj.probes]
        # theta, two_theta, gamma
        motor_list = np.array([-self.HKLToTheta(*HKL)
                               for HKL in self.params.HKL_list])
        self.motor_list_deg = motor_list
        self.motor_list_rad = motor_list * np.pi / 180
        
        # Using these redundant representations just for convenience
        self.thetas_deg, self.two_thetas_deg, self.gammas_deg = motor_list.T * 180 / np.pi
        self.thetas_rad, self.two_thetas_rad, self.gammas_rad = motor_list.T
        
        self.ptycho_scan_positions = self.getPtychoScanPositions(self.params.n_scan_positions,
                                                                 self.params.npix_scan_shift)
        self.simulations_per_peak = [SimulationPerBraggPeak(self.rhos[i], 
                                                            self.probes[i],
                                                            *self.motor_list_rad[i],
                                                            self.sample.params,
                                                            self.ptycho_scan_positions,
                                                            self.params.npix_det,
                                                            self.sample.x_trunc,
                                                            self.sample.y_trunc,
                                                            self.sample.z_trunc,
                                                            self.params.poisson_noise,
                                                            self.params.poisson_noise_level)
                                     for i in range(len(self.rhos))]
     
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
        y = np.ceil(np.arange(-nhalf, nhalf)).astype('int') * npix_scan_shift #* sample_pix_size
        x = np.ceil(np.arange(-nhalf, nhalf)).astype('int') * npix_scan_shift #* sample_pix_size
        Yscan, Xscan = np.meshgrid(y, x, indexing='ij')
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
        ang_sym = np.arcsin((np.sqrt(H**2 + K**2 + L**2) / 2) / (self.sample.params.lattice[0] / 
                                                                 self.params.wavelength))
        ang2_sym = 2 * ang_sym

        det_phi = np.arcsin(np.cos(ang_y) / np.sin(np.pi / 2 - ang_sym))
        temp = (np.sin(ang2_sym) * np.cos(det_phi)) ** 2 + np.cos(ang2_sym)**2

        theta = np.arcsin(np.cos(np.pi / 2 - ang_sym) / np.sin(ang_y))
        two_theta = np.arcsin(np.sin(ang2_sym) * np.cos(det_phi) / temp**0.5)
        gamma = np.arctan(np.sin(ang2_sym) * np.sin(det_phi) / temp**0.5)

        return np.array([theta, two_theta, gamma])
    
    
