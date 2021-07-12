import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from mbptycho.code.simulation import Simulation, reloadSimulation
from skimage.feature import register_translation
from skimage.restoration import unwrap_phase

import tensorflow_probability as tfp

class ReconstructionT:
    def __init__(self, simulation: Simulation,
                 batch_size: int = 50,
                 reconstruction_type: str = "displacement",
                 magnitudes_init: np.ndarray = None, 
                 ux_uy_init: np.ndarray = None, 
                 phases_init: np.ndarray = None,
                 unwrap_phase: bool = True,
                 shared_magnitudes: bool = True,
                 fit_displacement_to_phase_only: bool = False,
                 displacement_tv_reg_const: float = None):
        self.sim = simulation
        self.batch_size = batch_size
        self.unwrap_phase = unwrap_phase

        if reconstruction_type not in ["displacement", "phase", "alternating"]:
            raise ValueError("Supplied recons type is not supported.")
        self.reconstruction_type = reconstruction_type

        self.setPixelsAndPads()
        self.setGroundTruths()

        self.shared_magnitudes = shared_magnitudes
        self.displacement_tv_reg_const = displacement_tv_reg_const

        self.fit_displacement_to_phase_only = fit_displacement_to_phase_only
        if fit_displacement_to_phase_only and (phases_init is None or reconstruction_type!= 'displacement'):
            raise ValueError("Need to supply specify the 'displacement' fit and supply phases if we want to " +
                             "fit displacement to phases.")

        with tf.device('/gpu:0'):
            self.coords_t = tf.constant(self.sim.simulations_per_peak[0].nw_coords_stacked, dtype='float32')

            if not fit_displacement_to_phase_only:
                self.setInterpolationLimits()
                self.setPtychographyDataset()
                self.setProbesAndMasks()
                self.setLocationIndices()
                self.setSliceIndices()

            if not self.reconstruction_type == 'phase':
                self.setPhaseDisplacementInversionMatrix()

            # For the reconstructions, I am starting out with 1d variable arrays just for my own convenience
            # This is because my second order optimization algorithms require 1d variable arrays.
            # Starting directly with 2d variable arrays is fine.

            #self.constraint_fn = lambda x: tf.clip_by_value
            # I should be applying the constraint on the full
            if not fit_displacement_to_phase_only:
                self.constraint_fn = tf.nn.relu
                if shared_magnitudes:
                    size = self.npix_xy + self.sim.params.HKL_list.shape[0] - 1
                else:
                    size = self.npix_xy * self.sim.params.HKL_list.shape[0]

                if magnitudes_init is None:
                    magnitudes_init = np.ones(size)
                else:
                    if magnitudes_init.size != size:
                        raise ValueError("magnitude initialization supplied is not valid")

                self.magnitudes_v = tf.Variable(magnitudes_init,
                                                constraint = self.constraint_fn,
                                                dtype='float32',
                                                name='magnitudes')
            if self.reconstruction_type != "phases":
                if ux_uy_init is None:
                    ux_uy_init = np.ones(self.npix_xy * 2)
                self.ux_uy_2d_v = tf.Variable(ux_uy_init, dtype='float32', name='ux_uy')

            if phases_init is None:
                self.phases_v = tf.Variable(tf.zeros(self.npix_xy * self.sim.params.HKL_list.shape[0]),
                                            dtype='float32', name='phases')
            else:
                self.phases_v = tf.Variable(phases_init, dtype='float32', name='phases')

            if self.reconstruction_type != 'displacement' or self.fit_displacement_to_phase_only:
                self.unwrapped_phases_v = tf.Variable(self.getUnwrappedPhases(),
                                                          name='unwrapped_phases',
                                                          trainable=False)
            if self.reconstruction_type == 'alternating':
                self.proximal_const_v = tf.Variable(1.0, dtype='float32', trainable=False)
        if not fit_displacement_to_phase_only:
            self.setupMinibatch()
        self.optimizers = {}       
        self.iteration = tf.constant(0)
        self.objective_loss_per_iter = []
        

    def setGroundTruths(self):
        # These are the true values. Used for comparison and validation.
        pady0, padx0, nyvar, nxvar, nzvar = self.pady0, self.padx0, self.npix_y, self.npix_x, self.npix_z
        self.ux_test = self.sim.sample.Ux_trunc[..., nzvar // 2].copy()
        self.ux_test[~self.sim.sample.magnitudes_trunc_mask[..., nzvar // 2]] = 0
        self.ux_test = self.ux_test[pady0: pady0 + nyvar, padx0: padx0 + nxvar] / self.sim.sample.params.lattice[0]

        self.uy_test = self.sim.sample.Uy_trunc[..., nzvar // 2].copy()
        self.uy_test[~self.sim.sample.magnitudes_trunc_mask[...,nzvar//2]] = 0
        self.uy_test = self.uy_test[pady0: pady0 + nyvar, padx0: padx0 + nxvar] / self.sim.sample.params.lattice[0]

        self.rho_test = self.sim.sample.rhos[:,pady0: pady0 + nyvar, padx0: padx0 + nxvar, nzvar // 2]

        
    def setPixelsAndPads(self):
        self.npix_x = np.sum(np.sum(self.sim.sample.obj_mask_trunc, axis=(0,2)) > 0) 
        self.npix_y = np.sum(np.sum(self.sim.sample.obj_mask_trunc, axis=(1,2)) > 0) 
        self.npix_z = np.sum(np.sum(self.sim.sample.obj_mask_trunc, axis=(0,1)) > 0)
        
        self.npix_xy = self.npix_x * self.npix_y
        
        self.nyr, self.nxr, self.nzr = self.sim.rhos[0].shape
        self.pady0 = np.where(self.sim.sample.obj_mask_trunc.sum(axis=(1,2)))[0][0]
        self.padx0 = np.where(self.sim.sample.obj_mask_trunc.sum(axis=(0,2)))[0][0]
        self.padz0 = np.where(self.sim.sample.obj_mask_trunc.sum(axis=(0,1)))[0][0]
        
        
    def setInterpolationLimits(self):
        # These arrays indicate the limits of the interpolation before the bragg projection.
        self.interp_x_ref_min_t = tf.constant([self.sim.sample.y_trunc[0], 
                                               self.sim.sample.x_trunc[0],
                                               self.sim.sample.z_trunc[0]], dtype='float32')
        self.interp_x_ref_max_t = tf.constant([self.sim.sample.y_trunc[-1], 
                                               self.sim.sample.x_trunc[-1],
                                               self.sim.sample.z_trunc[-1]], dtype='float32')
    def setPtychographyDataset(self):
        # Ptychographic scan positions
        self.scan_coords_t = tf.constant(self.sim.ptycho_scan_positions, dtype='int64')

        # The diffraction patterns.
        # I am including all the diffraction patterns (for all the bragg peaks) as one giant dataset.
        # During the minibatch optimization process, I select the specified number (minibatch size)  of diffraction 
        # patterns randomly.
        diffraction_data_t = tf.constant(np.array([s.diffraction_patterns for s in self.sim.simulations_per_peak]),
                                         dtype='float32')**0.5
        self.diffraction_data_t = tf.reshape(diffraction_data_t, 
                                             [-1, *self.sim.simulations_per_peak[0].diffraction_patterns[0].shape])
        
    def setProbesAndMasks(self):
        # This is a workaround bc the different arrays for the different siulations might not necessarily 
        # have the same number of elements. Tensorflow doesn't handle that well. 
        # ------------------------------------------------------------------------------------------------
        # Since I have truncated probe structures for efficiency reasons, they are not of the same 3d shape
        # for the different bragg peaks.
        self.probes_all_t = tf.constant(np.concatenate([s.probe.flatten() for s in 
                                                        self.sim.simulations_per_peak]), 
                                        dtype='complex64')

        # After rotation (for the final projection), we can speed up the interpolation by only selecting
        # the rotated coordinates that lie within the unrotated object box.
        # The number of coordinates to interpolate can vary per bragg peak.
        self.coords_rotated_masked_t = tf.constant(np.concatenate([s.nw_rotated_masked_coords.T.flatten() 
                                                              for s in self.sim.simulations_per_peak]),
                                                    dtype='float32')
        self.rotation_mask_indices_t = tf.constant(np.concatenate([s.nw_rotation_mask_indices.flatten()
                                                              for s in self.sim.simulations_per_peak]),
                                              dtype='int64')
        
    def setLocationIndices(self):
        # These contain information about how to select the appropriate probe, etc.
        probes_iloc_all = []
        coords_iloc_all = []
        rotix_iloc_all = []
        for i, sim_per_peak in enumerate(self.sim.simulations_per_peak):
            if i == 0:
                s1 = 0
                s2 = 0
                s3 = 0
            else:
                s1 = probes_iloc_all[i-1][0] + probes_iloc_all[i-1][1]
                s2 = coords_iloc_all[i-1][0] + coords_iloc_all[i-1][1]
                s3 = rotix_iloc_all[i-1][0] + rotix_iloc_all[i-1][1]
            probes_iloc_all.append([s1, sim_per_peak.probe.size, *sim_per_peak.probe.shape])
            coords_iloc_all.append([s2, sim_per_peak.nw_rotated_masked_coords.T.size, 
                                    *sim_per_peak.nw_rotated_masked_coords.T.shape])
            rotix_iloc_all.append([s3, sim_per_peak.nw_rotation_mask_indices.size, 
                                   *sim_per_peak.nw_rotation_mask_indices.shape])
        self.probes_iloc_t = tf.constant(probes_iloc_all, dtype='int64')
        self.coords_iloc_t = tf.constant(coords_iloc_all, dtype='int64')
        self.rotix_iloc_t = tf.constant(rotix_iloc_all, dtype='int64')
        
    def setSliceIndices(self):

        # A lot of this slicing and dicing is aimed at avoiding expensive "roll" operations. 
        # I am reusing indices computed during the forward simulation for the recons as well.
        self.scan_rho_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.rho_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')
        self.scan_probe_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.probe_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')
        self.scan_proj_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.proj_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')

    def setPhaseDisplacementInversionMatrix(self):
        """To solve an overdetermined linear system Ax = b, we can use the relation:
        x = (A^T A)^(-1) A^T b.

        This function calculates and stores the matrix (A^T A)^(-1) A^T  for the unwrapped phases -> displacement
        transformation.
        """
        hkl_matrix = np.array(self.sim.params.HKL_list)[:,:2] # Ignoring the z components
        inversion_matrix = np.linalg.inv(hkl_matrix.T @ hkl_matrix) @ hkl_matrix.T / ( 2 * np.pi)
        self.displacement_invert_matrix_t = tf.constant(inversion_matrix, dtype='float32')

    @staticmethod
    def getStartStopIndicesFromSliceList(slice_list):
        indices_list = []
        for slice_y, slice_x in slice_list:
            sy1 = slice_y.start
            sy2 = slice_y.stop
            sx1 = slice_x.start
            sx2 = slice_x.stop
            indices_list.append([sy1, sy2, sx1, sx2])
        return indices_list
    
    def getMagnitudes2D(self, magnitudes_v):
        if self.shared_magnitudes:
            magnitudes_2d_profile = tf.reshape(magnitudes_v[:self.npix_xy], (self.npix_y, self.npix_x))
            magnitudes_scaling_vars = magnitudes_v[self.npix_xy:]
            magnitudes_scaling_all = tf.concat(([1.0], magnitudes_scaling_vars), axis=0)
            magnitudes_2d_all = magnitudes_2d_profile * magnitudes_scaling_all[:,None, None]
        else:
            magnitudes_2d_all = tf.reshape(magnitudes_v, (-1, self.npix_y, self.npix_x))
        return magnitudes_2d_all
    
    def getUxUy2d(self, ux_uy_2d_v):
        ux_uy_2d_reshaped_t = tf.reshape(ux_uy_2d_v, (2, self.npix_y, self.npix_x))
        ux_2d_t = ux_uy_2d_reshaped_t[0]
        uy_2d_t = ux_uy_2d_reshaped_t[1]
        return ux_2d_t, uy_2d_t

    def get2dPhasesFromNormalizedDisplacements(self, Ux_t, Uy_t, Uz_t):
        phases_all_t = []
        for H, K, L in self.sim.params.HKL_list:
            t1 = H * Ux_t + K * Uy_t + L * Uz_t
            phase_t = 2 * np.pi * t1
            phases_all_t.append(phase_t)
        phases_all_t = tf.stack(phases_all_t)
        return phases_all_t

    def get2dPhasesFromDisplacementVars(self):
        ux_2d_t, uy_2d_t = self.getUxUy2d(self.ux_uy_2d_v)
        uz_2d_t = tf.zeros_like(ux_2d_t)
        phases = self.get2dPhasesFromNormalizedDisplacements(ux_2d_t, uy_2d_t, uz_2d_t)
        return phases


    def get2dRhoFromNormalizedDisplacements(self, Ux_t, Uy_t, Uz_t, magnitudes_2d_t):
        """Calculate the magnitude and phase profile using the provided magnitudes and HKL list. 

        Sets the rhos for the sample class, and also returns the rhos.

        Assumes that the displacements are normalized w.r.t. to the lattice constant, i.e.
        Ux_t = Ux_t_actual / lattice[0]
        Uy_t = Uy_t_actual / lattice[1]
        Uz_t = Uz_t actual / lattice[2]

        Parameters
        ----------
        ...
        Returns
        -------
        """
        phases_all_t = self.get2dPhasesFromNormalizedDisplacements(Ux_t, Uy_t, Uz_t)
        rhos_all_t = tf.complex(magnitudes_2d_t * tf.cos(phases_all_t), 
                                magnitudes_2d_t * tf.sin(phases_all_t))
        return rhos_all_t
    
    def get3dRhoFromDisplacements(self, ux_uy_2d_v, magnitudes_v):
        magnitudes_2d_all_t = self.getMagnitudes2D(magnitudes_v)

        ux_2d_t, uy_2d_t = self.getUxUy2d(ux_uy_2d_v)
        uz_2d_t = tf.zeros_like(ux_2d_t)

        rho_2d_cmplx_all_t = self.get2dRhoFromNormalizedDisplacements(ux_2d_t, uy_2d_t, uz_2d_t, magnitudes_2d_all_t)
        rho_3d_all_t = tf.ones(self.npix_z, dtype='complex64')[None,None,None, :] * rho_2d_cmplx_all_t[:, :,:, None]
        rho_3d_bordered_all_t = tf.pad(rho_3d_all_t, [[0,0], 
                                                      [self.pady0, self.nyr - self.pady0 - self.npix_y],
                                                      [self.padx0, self.nxr - self.padx0 - self.npix_x],
                                                      [self.padz0, self.nzr - self.padz0 - self.npix_z]])
        return rho_3d_bordered_all_t
    
    def get2dRhoFromPhases(self, phases_v, magnitudes_v):
        magnitudes_2d_all_t = self.getMagnitudes2D(magnitudes_v)
        phases_2d_t = tf.reshape(phases_v, [-1, self.npix_y, self.npix_x])
        rho_2d_cmplx_all_t = tf.complex(magnitudes_2d_all_t * tf.math.cos(phases_2d_t),
                                        magnitudes_2d_all_t * tf.math.sin(phases_2d_t))
        return rho_2d_cmplx_all_t
    
    def get3dRhoFromPhases(self, phases_v, magnitudes_v):
        rho_2d_cmplx_all_t = self.get2dRhoFromPhases(phases_v, magnitudes_v)
        rho_3d_all_t = tf.ones(self.npix_z, dtype='complex64')[None,None,None, :] * rho_2d_cmplx_all_t[:, :,:, None]
        rho_3d_bordered_all_t = tf.pad(rho_3d_all_t, [[0,0], 
                                                      [self.pady0, self.nyr - self.pady0 - self.npix_y],
                                                      [self.padx0, self.nxr - self.padx0 - self.npix_x],
                                                      [self.padz0, self.nzr - self.padz0 - self.npix_z]])
        return rho_3d_bordered_all_t
        
    
    def getNumpyOutputs(self):
        outs = {}
        if not self.fit_displacement_to_phase_only:
            magnitudes_2d_all_t = self.getMagnitudes2D(self.magnitudes_v)
            outs = {'magnitudes': magnitudes_2d_all_t.numpy()}

        if self.reconstruction_type != "phase":
            ux_2d_t, uy_2d_t = self.getUxUy2d(self.ux_uy_2d_v)
            uz_2d_t = tf.zeros_like(ux_2d_t)


            outs['ux'] = ux_2d_t.numpy()
            outs['uy'] = uy_2d_t.numpy()

            if not self.fit_displacement_to_phase_only:
                rho_2d_cmplx_all_t = self.get2dRhoFromNormalizedDisplacements(ux_2d_t, uy_2d_t, uz_2d_t,
                                                                              magnitudes_2d_all_t)
                outs['rho_2d_disp'] = rho_2d_cmplx_all_t.numpy()

        if self.reconstruction_type != "displacement":
            rho_2d_cmplx_all_t = self.get2dRhoFromPhases(self.phases_v, self.magnitudes_v)
            outs['rho_2d_phase'] = rho_2d_cmplx_all_t.numpy()

        return outs
    
        
    
    def setupMinibatch(self):
        
        # the total number of diffraction patterns is len(sm.ptycho_scan_positions) * len(sm.params.HKL_list)
        # We select a minibatch from there.

        self.n_diffraction_patterns = len(self.sim.ptycho_scan_positions) * len(self.sim.params.HKL_list)
        self.n_minibatch_per_epoch = self.n_diffraction_patterns // self.batch_size
        dataset = tf.data.Dataset.range(self.n_diffraction_patterns)
        dataset = dataset.shuffle(self.n_diffraction_patterns)
        dataset = dataset.repeat()

        dataset_batch = dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', 2))
        self.iterator = iter(self.dataset_batch)

        # I am using an extra variable here to ensure that the minibatch is only updated exactly when
        # I want. 
        # This variable is not an optimization variable. 
        self.batch_train_input_v = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int64), trainable=False)
    
    
    def getProjFtT(self, rho_3d_bordered_all_t, scan_index):
        """Output the projected diffraction pattern at the input scan index.

        Parameters
        ----------
        scan_index_t : tensor(int)
            This indicates the location of the selected diffraction pattern in the overall 
            large array containing all the diffraction patterns, for all the bragg peaks.
        Returns
        -------
        out_t : tensor(float32)
            This is the predicted diffraction pattern.
        """
        # 

        # Identifying the appropriate bragg peak and ptychographic scan position for the given
        # scan index
        bragg_index = scan_index // len(self.sim.ptycho_scan_positions)
        scan_coord_index = scan_index % len(self.sim.ptycho_scan_positions)
        scan_coord_t = self.scan_coords_t[scan_coord_index]

        # the y and x positions for the scan
        pcy = scan_coord_t[0]
        pcx = scan_coord_t[1]

        # again, a lot the the slicing and dicing 
        rho_slice = self.scan_rho_slice_indices_t[bragg_index, scan_coord_index]
        proj_slice = self.scan_proj_slice_indices_t[bragg_index, scan_coord_index]
        probe_slice = self.scan_probe_slice_indices_t[bragg_index, scan_coord_index]

        iloc1 = self.probes_iloc_t[bragg_index]
        # Slicing and reshaping to get the probe
        probe_this = tf.reshape(self.probes_all_t[iloc1[0]: iloc1[0] + iloc1[1]], iloc1[2:])

        iloc2 = self.coords_iloc_t[bragg_index]
        # getting the coordinates that we want to interpolate at.
        coords_rotated_masked_this = tf.reshape(self.coords_rotated_masked_t[iloc2[0]: iloc2[0] + iloc2[1]],
                                                iloc2[2:])

        iloc3 = self.rotix_iloc_t[bragg_index]
        rotation_mask_indices_this = tf.reshape(self.rotation_mask_indices_t[iloc3[0]: iloc3[0] + iloc3[1]],
                                                iloc3[2:])[:,None]

        # Getting the slice of the object numerical window that the probe interacts with and calculating
        # the interaction.
        rho_this = rho_3d_bordered_all_t[bragg_index]
        field_view_t = (rho_this[rho_slice[0]: rho_slice[1], rho_slice[2]:rho_slice[3]] 
                        * probe_this[probe_slice[0]: probe_slice[1], probe_slice[2]: probe_slice[3]])

        # setting the field everywhere else to 0.
        field_view_padded_t = tf.pad(field_view_t, [[rho_slice[0], 200-rho_slice[1]],
                                                    [rho_slice[2], 200 - rho_slice[3]],
                                                    [0, 0]], 
                                    mode='constant')

        # iinterpolation for the rotated coordinates that lie within the unrotated object.
        rot_field_reals_t = tfp.math.batch_interp_regular_nd_grid(x=coords_rotated_masked_this,
                                                                  x_ref_min=self.interp_x_ref_min_t,
                                                                  x_ref_max=self.interp_x_ref_max_t,
                                                                  y_ref=tf.math.real(field_view_padded_t),
                                                                  axis=0, fill_value=0)
        rot_field_imag_t = tfp.math.batch_interp_regular_nd_grid(x=coords_rotated_masked_this,
                                                                 x_ref_min=self.interp_x_ref_min_t,
                                                                 x_ref_max=self.interp_x_ref_max_t,
                                                                 y_ref=tf.math.imag(field_view_padded_t),
                                                                 axis=0, fill_value=0)

        # since we only interpolate a small selection of the rotated coordinates, we need to 
        # fill the rest of the coordinates with 0.
        field_rotated_real_t = tf.scatter_nd(indices=rotation_mask_indices_this,
                                             updates=tf.reshape(rot_field_reals_t, [-1]), 
                                             shape=[tf.size(rho_this)])#field_this.size])
        field_rotated_imag_t = tf.scatter_nd(indices=rotation_mask_indices_this,
                                             updates=tf.reshape(rot_field_imag_t, [-1]), 
                                             shape=[tf.size(rho_this)])#field_this.size])
        field_rotated_t = tf.complex(field_rotated_real_t, field_rotated_imag_t)

        field_rotated_t = tf.reshape(field_rotated_t, tf.shape(rho_this))

        # Project along the (rotated) z direction to get the exit wave.
        projection_t = tf.reduce_sum(field_rotated_t, axis=2)
        projection_t = projection_t[proj_slice[0]:proj_slice[1], proj_slice[2]:proj_slice[3]]
        proj_ft_t = tf.signal.fft2d(projection_t)
        out_t =  tf.abs(proj_ft_t)
        return out_t
    
    #@tf.function
    def _batchPredictFromDisplacementsMagnitudes(self, ux_uy_2d_v, magnitudes_v):
        rho_3d_bordered_all_t = self.get3dRhoFromDisplacements(ux_uy_2d_v, magnitudes_v)
        
        # Getting the predicted diffraction data. Using a map is faster than a for loop.
        proj_map_fn = lambda indx: self.getProjFtT(rho_3d_bordered_all_t, indx)
        batch_predictions_t = tf.map_fn(proj_map_fn, 
                                        self.batch_train_input_v,
                                        fn_output_signature=[])
        out = tf.reshape(batch_predictions_t, [-1])
        return out
    
    #@tf.function
    def _batchPredictFromPhasesMagnitudes(self, phases_v, magnitudes_v):
        rho_3d_bordered_all_t = self.get3dRhoFromPhases(phases_v, magnitudes_v)
        # Getting the predicted diffraction data. Using a map is faster than a for loop.
        proj_map_fn = lambda indx: self.getProjFtT(rho_3d_bordered_all_t, indx)
        batch_predictions_t = tf.map_fn(proj_map_fn, 
                                        self.batch_train_input_v,
                                        dtype=tf.float32)
        out = tf.reshape(batch_predictions_t, [-1])
        return out
    
    def displacementPredictFn(self, ux_uy_2d_v):
        return self._batchPredictFromDisplacementsMagnitudes(ux_uy_2d_v, self.magnitudes_v)
    
    def magnitudesPredictFn(self, magnitudes_v):
        if self.reconstruction_type == "displacement":
            return self._batchPredictFromDisplacementsMagnitudes(self.ux_uy_2d_v, magnitudes_v)
        else:
            return self._batchPredictFromPhasesMagnitudes(self.phases_v, magnitudes_v)
    
    #@tf.function
    def loss_fn(self, predicted_data_t):
        batch_diff_data_t = tf.reshape(tf.gather(self.diffraction_data_t, self.batch_train_input_v), [-1])
        return 0.5 * tf.reduce_mean((batch_diff_data_t - predicted_data_t)**2)
    
    def genNewBatch(self):
        batch_indices = self.batch_train_input_v.assign(self.iterator.next())
        # Selecting the measured diffraction data for the minibatch
        #self.batch_diff_data_t = 
        return batch_indices
    
    def setDisplacementAdamOptimizer(self, learning_rate):
        if self.fit_displacement_to_phase_only:
            raise ValueError("Cannot set adam optimizer when fitting the displacemnt to the phase. Use direct inversion.")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['displacement'] = {'learning_rate': lr_var,
                                           'optimizer': tf.keras.optimizers.Adam(lr_var),
                                           'var': self.ux_uy_2d_v}
    
    def setMagnitudeAdamOptimizer(self, learning_rate):
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['magnitude'] = {'learning_rate': lr_var,
                                        'optimizer': tf.keras.optimizers.Adam(lr_var),
                                        'var': self.magnitudes_v}

    def setPhaseAdamOptimizer(self, learning_rate):
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['phase'] = {'learning_rate': lr_var,
                                    'optimizer': tf.keras.optimizers.Adam(lr_var),
                                    'var': self.phases_v}

    def setPhaseSGDOptimizer(self, learning_rate):
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['phase'] = {'learning_rate': lr_var,
                                    'optimizer': tf.keras.optimizers.SGD(lr_var),
                                    'var': self.phases_v}

    
    def tv_loss_fn(self):
        ux_2d_t, uy_2d_t = self.getUxUy2d(self.ux_uy_2d_v)
        tvx = tf.image.total_variation(ux_2d_t[...,None])
        tvy = tf.image.total_variation(uy_2d_t[...,None])
        tv_total = self.displacement_tv_reg_const * (tvx + tvy)
        return tv_total
    
    def data_fit_fn(self):
        if self.reconstruction_type == 'displacement':
            return self.loss_fn(self._batchPredictFromDisplacementsMagnitudes(self.ux_uy_2d_v, self.magnitudes_v))
        else:
            return self.loss_fn(self._batchPredictFromPhasesMagnitudes(self.phases_v, self.magnitudes_v))
    
    def objective_fn(self):
        if self.displacement_tv_reg_const is not None:
            return self.data_fit_fn() + self.tv_loss_fn()
        return self.data_fit_fn()

    #def constraint_fit_fn(self):
    #    phases_from_displacements = tf.reshape(self.get2dPhasesFromDisplacementVars(), [-1])
    #    #phases_unwrapped = self.getUnwrappedPhases()
    #    #if not self.fit_displacement_to_phase_only:
    #    #    self.unwrapped_phases_v.assign(self.getUnwrappedPhases())
    #    return 0.5 * tf.reduce_sum((phases_from_displacements - self.unwrapped_phases_v)**2)

    def proximal_fit_fn(self):
        phases_from_displacements = tf.reshape(self.get2dPhasesFromDisplacementVars(), [-1])
        #phases_wrapped = phases_from_displacements
        phases_wrapped = tf.math.angle(tf.complex(tf.math.cos(phases_from_displacements),
                                                  tf.math.sin(phases_from_displacements)))
        return 0.5 * self.proximal_const_v * tf.reduce_sum((tf.reshape(phases_wrapped, [-1]) - self.phases_v)**2)

    def alternating_objective_fn(self):
        return self.objective_fn() + self.proximal_fit_fn()

    def convertPhaseToDisplacement(self, phase_flat):
        phases = tf.reshape(phase_flat, (self.displacement_invert_matrix_t.shape[1], -1))
        ux_uy_2d = self.displacement_invert_matrix_t @ phases
        self.ux_uy_2d_v.assign(tf.reshape(ux_uy_2d, [-1]))

    @tf.function
    def optimizersMinimize(self, iters_before_check: tf.Tensor, inner_iters: tf.Tensor):
        #if self.reconstruction_type == 'alternating':
        #    raise ValueError("Cannot use adamMinimize function for alternating optimization")
        objective_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        losses_stack = []
        for i in range(iters_before_check):
            for in_it in range(inner_iters):
                self.genNewBatch()
                for key, opt in self.optimizers.items():
                    if self.reconstruction_type == 'alternating':
                        if key == 'phase':
                            opt['optimizer'].minimize(self.alternating_objective_fn, opt['var'])
                            continue
                        elif key == 'displacement':
                            continue
                    opt['optimizer'].minimize(self.objective_fn, opt['var'])

            if self.reconstruction_type == 'alternating':
                self.unwrapped_phases_v.assign(self.getUnwrappedPhases())
                self.convertPhaseToDisplacement(self.unwrapped_phases_v)
                proximal_loss = self.proximal_fit_fn()
                losses_stack.append(proximal_loss)
                #phases_from_displacements = tf.reshape(self.get2dPhasesFromDisplacementVars(), [-1])
                #phases_wrapped = tf.math.angle(tf.complex(tf.math.cos(phases_from_displacements),
                #                                          tf.math.sin(phases_from_displacements)))
                #self.phases_v.assign(tf.reshape(phases_wrapped, [-1]))
                #for disp_in_it in range(displacement_inner_iters):
                #    opt = self.optimizers['displacement']
                #    opt['optimizer'].minimize(self.constraint_fit_fn, opt['var'])

            objective_loss = self.objective_fn()
            losses_stack.append(objective_loss)

            objective_array = objective_array.write(i, losses_stack)

        return objective_array.stack()

    def checkErrors(self, losses_stack=None):
        outs = self.getNumpyOutputs()
        obj_mask = self.sim.sample.magnitudes_trunc_mask
        #obj_mask = self.sim.sample.obj_mask_trunc_no_pad
        err_print_append = ""
        if self.reconstruction_type != 'phase':
            pady0, padx0, nyvar, nxvar, nzvar = self.pady0, self.padx0, self.npix_y, self.npix_x, self.npix_z

            outs['ux'][
                ~(obj_mask[pady0: pady0 + nyvar, padx0: padx0 + nxvar, nzvar // 2])] = 0
            outs['uy'][
                ~(obj_mask[pady0: pady0 + nyvar, padx0: padx0 + nxvar, nzvar // 2])] = 0

            ux_test = self.ux_test - self.ux_test.mean()
            ux_out = outs['ux'] - outs['ux'].mean()
            diff_sumx = np.abs(ux_test - ux_out).sum()

            uy_test = self.uy_test - self.uy_test.mean()
            uy_out = outs['uy'] - outs['uy'].mean()
            diff_sumy = np.abs(uy_test - uy_out).sum()
            rollx, errx, phasex = register_translation(self.ux_test - self.ux_test.mean(),
                                                       outs['ux'] - outs['ux'].mean(), upsample_factor=1)
            rolly, erry, phasey = register_translation(self.uy_test - self.uy_test.mean(),
                                                       outs['uy'] - outs['uy'].mean(), upsample_factor=1)
            err_print_append = f" err_ux {errx:4.3g} diffsum_ux {diff_sumx: 4.3g} err_uy {erry:4.3g} diffsum_uy {diff_sumy:4.3g}"

        rho_err_print_append = ""
        if not self.fit_displacement_to_phase_only:
            rho_err_print_append = "err_rho "
            for indx, rho_test in enumerate(self.rho_test):

                if self.reconstruction_type == 'displacement':
                    phases = np.angle(outs['rho_2d_disp'])
                else:
                    phases = outs['rho_2d_phase']
                rollr, errr, phaser = register_translation(rho_test, phases[indx], upsample_factor=10)
                rollr, errr, phaser = register_translation(rho_test, phases[indx] * np.exp(1j * phaser),
                                                           upsample_factor=10)

                rho_err_print_append += f"{errr:4.3g} "
        itn = self.iteration.numpy()
        print_out = f"Iter {itn}"
        if not self.fit_displacement_to_phase_only:
            if len(self.objective_loss_per_iter) ==0:
                loss = self.objective_fn()
            else:
                loss = self.objective_loss_per_iter[-1]
            print_out += f" obj_loss {loss:4.3g} "
            if losses_stack is not None:
                print_out += " other_losses "
                for l in losses_stack[:-1]:
                    print(l)
                    print_out += f" {l:4.3g} "
        print(print_out + rho_err_print_append + err_print_append)


    def minimizeAndCheckErrors(self, n_iterations, check_frequency=10, inner_iters=1):

        n_outer = n_iterations // check_frequency

        for j in range(n_outer):
            if self.fit_displacement_to_phase_only:
                self.convertPhaseToDisplacement(self.unwrapped_phases_v)
                self.iteration += 1
                self.checkErrors()
                break
            else:
                print('outer', j)
                losses_stack = self.optimizersMinimize(tf.constant(check_frequency),
                                                       tf.constant(inner_iters)).numpy()
                self.objective_loss_per_iter = np.append(self.objective_loss_per_iter, losses_stack[:, -1])

                self.iteration += check_frequency
                self.checkErrors(losses_stack[-1])


    def getUnwrappedPhases(self):
        if self.reconstruction_type == 'displacement' and not self.fit_displacement_to_phase_only:
            raise NotImplementedError("Unwrapping phases only works for the phase recons procedure.")

        def _unwrap_phases(phases):
            phases_reshaped = np.reshape(phases, [-1, self.npix_y, self.npix_x])
            unwrapped = []
            for p in phases_reshaped:
                p_new = unwrap_phase(p)
                unwrapped.append(p_new)
            return np.array(unwrapped, 'float32').reshape(-1)
        if self.unwrap_phase:
            phases_unwrapped = tf.py_function(_unwrap_phases, inp=[self.phases_v],
                                              Tout=[tf.float32])
        else:
            phases_unwrapped = tf.reshape(self.phases_v, [-1])

        return tf.reshape(phases_unwrapped, [-1])
        