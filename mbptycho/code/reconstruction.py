import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from mbptycho.code.simulation import Simulation, reloadSimulation
from skimage.feature import register_translation

import tensorflow_probability as tfp


class ReconstructionT:
    def __init__(self, simulation: Simulation,
                 batch_size: int = 50, 
                 reconstruct_phases_only: bool = False,
                 amplitudes_init: np.ndarray = None, 
                 ux_uy_init: np.ndarray = None, 
                 phases_init: np.ndarray = None,
                 displacement_tv_reg_const: float = None):
        self.sim = simulation
        self.batch_size = batch_size
        self.reconstruct_phases_only = reconstruct_phases_only
        self.setPixelsAndPads()
        self.setGroundTruths()
        
        self.displacement_tv_reg_const = displacement_tv_reg_const
        
        with tf.device('/gpu:0'):
            self.coords_t = tf.constant(self.sim.simulations_per_peak[0].nw_coords_stacked, dtype='float32')
            
            self.setInterpolationLimits()
            self.setPtychographyDataset()
            self.setProbesAndMasks()
            self.setLocationIndices()
            self.setSliceIndices()

            # For the reconstructions, I am starting out with 1d variable arrays just for my own convenience
            # This is because my second order optimization algorithms require 1d variable arrays.
            # Starting directly with 2d variable arrays is fine.

            self.constraint_fn = lambda x: tf.clip_by_value(x, 0, 1.0)
            if amplitudes_init is None:
                amplitudes_init = np.ones(self.npix_xy + self.sim.params.HKL_list.shape[0] - 1)
            self.amplitudes_v = tf.Variable(amplitudes_init,  
                                            constraint = self.constraint_fn, 
                                            dtype='float32')
            if not self.reconstruct_phases_only: 
                if ux_uy_init is None:
                    ux_uy_init = np.ones(self.npix_xy * 2)
                self.ux_uy_2d_v = tf.Variable(ux_uy_init, dtype='float32')
            else:
                self.phases_v = tf.Variable(tf.zeros(self.npix_xy * self.sim.params.HKL_list.shape[0]),
                                            dtype='float32')
        
        self.setupMinibatch()
        self.optimizers = {}       
        self.iteration = tf.constant(0)
        self.loss_per_iter = []
        

    def setGroundTruths(self):
        # These are the true values. Used for comparison and validation.
        pady0, padx0, nyvar, nxvar, nzvar = self.pady0, self.padx0, self.npix_y, self.npix_x, self.npix_z
        self.ux_test = self.sim.sample.Ux_trunc[..., nzvar // 2].copy()
        self.ux_test[~self.sim.sample.amplitudes_trunc_mask[..., nzvar // 2]] = 0
        self.ux_test = self.ux_test[pady0: pady0 + nyvar, padx0: padx0 + nxvar] / self.sim.sample.params.lattice[0]

        self.uy_test = self.sim.sample.Uy_trunc[..., nzvar // 2].copy()
        self.uy_test[~self.sim.sample.amplitudes_trunc_mask[...,nzvar//2]] = 0
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
                                         dtype='float32')
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
        # I am reusing indices computed during the forward simulation for the reconstruction as well.
        self.scan_rho_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.rho_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')
        self.scan_probe_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.probe_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')
        self.scan_proj_slice_indices_t = tf.constant([self.getStartStopIndicesFromSliceList(s.proj_slices)
                                                for s in self.sim.simulations_per_peak], dtype='int64')
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
    
    def getAmplitudes2D(self, amplitudes_v):
        amplitudes_2d_profile = tf.reshape(amplitudes_v[:self.npix_xy], (self.npix_y, self.npix_x))
        amplitudes_scaling_vars = amplitudes_v[self.npix_xy:]
        amplitudes_scaling_all = tf.concat(([1.0], amplitudes_scaling_vars), axis=0)
        amplitudes_2d_all = amplitudes_2d_profile * amplitudes_scaling_all[:,None, None]
        return amplitudes_2d_all
    
    def getUxUy2d(self, ux_uy_2d_v):
        ux_uy_2d_reshaped_t = tf.reshape(ux_uy_2d_v, (2, self.npix_y, self.npix_x))
        ux_2d_t = ux_uy_2d_reshaped_t[0]
        uy_2d_t = ux_uy_2d_reshaped_t[1]
        return ux_2d_t, uy_2d_t
    
    def get2dRhoFromNormalizedDisplacements(self, Ux_t, Uy_t, Uz_t, amplitudes_2d_t):
        """Calculate the amplitude and phase profile using the provided amplitudes and HKL list. 

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
        phases_all_t = []
        for H, K, L in self.sim.params.HKL_list:
            t1 = H * Ux_t + K * Uy_t  +  L * Uz_t 
            phase_t = 2 * np.pi * t1
            phases_all_t.append(phase_t)
        phases_all_t = tf.stack(phases_all_t)
        rhos_all_t = tf.complex(amplitudes_2d_t * tf.cos(phases_all_t), 
                                amplitudes_2d_t * tf.sin(phases_all_t))
        return rhos_all_t
    
    def get3dRhoFromDisplacements(self, ux_uy_2d_v, amplitudes_v):
        amplitudes_2d_all_t = self.getAmplitudes2D(amplitudes_v)

        ux_2d_t, uy_2d_t = self.getUxUy2d(ux_uy_2d_v)
        uz_2d_t = tf.zeros_like(ux_2d_t)

        rho_2d_cmplx_all_t = self.get2dRhoFromNormalizedDisplacements(ux_2d_t, uy_2d_t, uz_2d_t, amplitudes_2d_all_t)
        rho_3d_all_t = tf.ones(self.npix_z, dtype='complex64')[None,None,None, :] * rho_2d_cmplx_all_t[:, :,:, None]
        rho_3d_bordered_all_t = tf.pad(rho_3d_all_t, [[0,0], 
                                                      [self.pady0, self.nyr - self.pady0 - self.npix_y],
                                                      [self.padx0, self.nxr - self.padx0 - self.npix_x],
                                                      [self.padz0, self.nzr - self.padz0 - self.npix_z]])
        return rho_3d_bordered_all_t
    
    def get2dRhoFromPhases(self, phases_v, amplitudes_v):
        amplitudes_2d_all_t = self.getAmplitudes2D(amplitudes_v)
        phases_2d_t = tf.reshape(phases_v, [-1, self.npix_y, self.npix_x])
        rho_2d_cmplx_all_t = tf.complex(amplitudes_2d_all_t * tf.math.cos(phases_2d_t),
                                        amplitudes_2d_all_t * tf.math.sin(phases_2d_t))
        return rho_2d_cmplx_all_t
    
    def get3dRhoFromPhases(self, phases_v, amplitudes_v):
        rho_2d_cmplx_all_t = self.get2dRhoFromPhases(phases_v, amplitudes_v)
        rho_3d_all_t = tf.ones(self.npix_z, dtype='complex64')[None,None,None, :] * rho_2d_cmplx_all_t[:, :,:, None]
        rho_3d_bordered_all_t = tf.pad(rho_3d_all_t, [[0,0], 
                                                      [self.pady0, self.nyr - self.pady0 - self.npix_y],
                                                      [self.padx0, self.nxr - self.padx0 - self.npix_x],
                                                      [self.padz0, self.nzr - self.padz0 - self.npix_z]])
        return rho_3d_bordered_all_t
        
    
    def getNumpyOutputs(self):
        amplitudes_2d_all_t = self.getAmplitudes2D(self.amplitudes_v)
        if not self.reconstruct_phases_only:
            ux_2d_t, uy_2d_t = self.getUxUy2d(self.ux_uy_2d_v)
            uz_2d_t = tf.zeros_like(ux_2d_t)

            rho_2d_cmplx_all_t = self.get2dRhoFromNormalizedDisplacements(ux_2d_t, uy_2d_t, uz_2d_t, amplitudes_2d_all_t)
            outs = [x.numpy() for x in [amplitudes_2d_all_t, ux_2d_t, uy_2d_t, rho_2d_cmplx_all_t]]
        
        else:
            rho_2d_cmplx_all_t = self.get2dRhoFromPhases(self.phases_v, self.amplitudes_v)
            outs = [x.numpy() for x in [amplitudes_2d_all_t, rho_2d_cmplx_all_t]]
        
        return outs
    
        
    
    def setupMinibatch(self):
        
        # the total number of diffraction patterns is len(sm.ptycho_scan_positions) * len(sm.params.HKL_list)
        # We select a minibatch from there.
        dataset = tf.data.Dataset.range(len(self.sim.ptycho_scan_positions) * len(self.sim.params.HKL_list))
        dataset = dataset.shuffle(len(self.sim.ptycho_scan_positions) * len(self.sim.params.HKL_list))
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
    
    @tf.function
    def _batchPredictFromDisplacementsAmplitudes(self, ux_uy_2d_v, amplitudes_v):
        rho_3d_bordered_all_t = self.get3dRhoFromDisplacements(ux_uy_2d_v, amplitudes_v)
        
        # Getting the predicted diffraction data. Using a map is faster than a for loop.
        proj_map_fn = lambda indx: self.getProjFtT(rho_3d_bordered_all_t, indx)
        batch_predictions_t = tf.map_fn(proj_map_fn, 
                                        self.batch_train_input_v,
                                        dtype=tf.float32)
        out = tf.reshape(batch_predictions_t, [-1])
        return out
    
    @tf.function
    def _batchPredictFromPhasesAmplitudes(self, phases_v, amplitudes_v):
        rho_3d_bordered_all_t = self.get3dRhoFromPhases(phases_v, amplitudes_v)
        # Getting the predicted diffraction data. Using a map is faster than a for loop.
        proj_map_fn = lambda indx: self.getProjFtT(rho_3d_bordered_all_t, indx)
        batch_predictions_t = tf.map_fn(proj_map_fn, 
                                        self.batch_train_input_v,
                                        dtype=tf.float32)
        out = tf.reshape(batch_predictions_t, [-1])
        return out
    
    def displacementPredictFn(self, ux_uy_2d_v):
        return self._batchPredictFromDisplacementsAmplitudes(ux_uy_2d_v, self.amplitudes_v)
    
    def amplitudesPredictFn(self, amplitudes_v):
        if self.reconstruct_phases_only:
            return self._batchPredictFromPhasesAmplitudes(self.phases_v, amplitudes_v)
        else:
            return self._batchPredictFromDisplacementsAmplitudes(self.ux_uy_2d_v, amplitudes_v)
    
    @tf.function
    def loss_fn(self, predicted_data_t):
        batch_diff_data_t = tf.reshape(tf.gather(self.diffraction_data_t, self.batch_train_input_v), [-1])
        return 0.5 * tf.reduce_mean((batch_diff_data_t - predicted_data_t)**2)
    
    def genNewBatch(self):
        batch_indices = self.batch_train_input_v.assign(self.iterator.next())
        # Selecting the measured diffraction data for the minibatch
        #self.batch_diff_data_t = 
        return batch_indices
    
    def setDisplacementAdamOptimizer(self, learning_rate):
        self.optimizers['displacement'] = {'optimizer': tf.keras.optimizers.Adam(learning_rate),
                                           'var': self.ux_uy_2d_v}
    
    def setAmplitudeAdamOptimizer(self, learning_rate):
        self.optimizers['amplitude'] = {'optimizer': tf.keras.optimizers.Adam(learning_rate),
                                        'var': self.amplitudes_v}
    
    def setPhaseAdamOptimizer(self, learning_rate):
        self.optimizers['phase'] = {'optimizer': tf.keras.optimizers.Adam(learning_rate),
                                    'var': self.phases_v}
    
    def tv_loss_fn(self):
        ux_2d_t, uy_2d_t = self.getUxUy2d(self.ux_uy_2d_v)
        tvx = tf.image.total_variation(ux_2d_t[...,None])
        tvy = tf.image.total_variation(uy_2d_t[...,None])
        tv_total = self.displacement_tv_reg_const * (tvx + tvy)
        return tv_total
    
    def fit_fn(self):
        if self.reconstruct_phases_only:
            return self.loss_fn(self._batchPredictFromPhasesAmplitudes(self.phases_v, self.amplitudes_v))
        else:
            return self.loss_fn(self._batchPredictFromDisplacementsAmplitudes(self.ux_uy_2d_v, self.amplitudes_v))
    
    def objective_fn(self):
        if self.displacement_tv_reg_const is not None:
            return self.fit_fn() + self.tv_loss_fn()
        return self.fit_fn()
            
    @tf.function
    def adamMinimize(self, n_inner: tf.Tensor):
        objective_array = tf.TensorArray(tf.float32, size=0,dynamic_size=True, clear_after_read=False)
        for i in range(n_inner):
            self.genNewBatch()
            
            #objective_fn = lambda: self.loss_fn(self._batch_predict_fn(self.ux_uy_2d_v, self.amplitudes_v))
            for key, opt in self.optimizers.items():
                opt['optimizer'].minimize(self.objective_fn, opt['var'])
            
            objective = self.objective_fn()
            objective_array = objective_array.write(i, objective)
        return objective_array.stack()
    
    def minimizeAndCheckErrors(self, n_iterations, check_frequency=10):
        
        n_outer = n_iterations // check_frequency
        for j in range(n_outer):
            losses_stack = self.adamMinimize(tf.constant(check_frequency))
           
            self.loss_per_iter = np.concatenate((self.loss_per_iter, losses_stack.numpy()))
            self.iteration += check_frequency
                                                
            if self.reconstruct_phases_only:
                amplitudes_2d_out, rho_out = self.getNumpyOutputs()
            else:
                amplitudes_2d_out, ux_out, uy_out, rho_out = self.getNumpyOutputs()
            
            err_print_append = ""
            if  not self.reconstruct_phases_only:
                pady0, padx0, nyvar, nxvar, nzvar = self.pady0, self.padx0, self.npix_y, self.npix_x, self.npix_z
            
                ux_out[~(self.sim.sample.amplitudes_trunc_mask[pady0: pady0 + nyvar, padx0: padx0 + nxvar, nzvar//2])] = 0
                uy_out[~(self.sim.sample.amplitudes_trunc_mask[pady0: pady0 + nyvar, padx0: padx0 + nxvar, nzvar//2])] = 0

                rollx, errx, phasex = register_translation(self.ux_test, ux_out, upsample_factor=2)
                rolly, erry, phasey = register_translation(self.uy_test, uy_out, upsample_factor=2)
                err_print_append = f" err_ux {errx:4.3g} err_uy {erry:4.3g}"
            
            rho_err_print_append = "err_rho "
            for indx, rho_test in enumerate(self.rho_test):
                rollr, errr, phaser = register_translation(rho_test, rho_out[indx], upsample_factor=10)
                rollr, errr, phaser = register_translation(rho_test, rho_out[indx] * np.exp(1j * phaser), upsample_factor=10)
                
                rho_err_print_append += f"{errr:4.3g} "
            
            itn, lossval = self.iteration.numpy(), self.loss_per_iter[-1]
            print(f"Iter {itn} floss {lossval:4.3g} " + rho_err_print_append + err_print_append)
        
        
        