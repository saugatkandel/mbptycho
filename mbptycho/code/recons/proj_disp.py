import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from mbptycho.code.simulation import Simulation, reloadSimulation
from skimage.feature import register_translation
from skimage.restoration import unwrap_phase

import tensorflow_probability as tfp
from mbptycho.code.recons.reconstruction_old import  ReconstructionT

class ProjectedDisplacementReconstructionT(ReconstructionT):
    def __init__(self, simulation: Simulation,
                 batch_size: int = 50,
                 magnitudes_init: np.ndarray = None, 
                 ux_uy_init: np.ndarray = None, 
                 phases_init: np.ndarray = None,
                 shared_magnitudes: bool = True,
                 displacement_tv_reg_const: float = None):
        self.sim = simulation
        self.batch_size = batch_size
        self.reconstruction_type = "alternating"
        self.unwrap_phase = True
        self.fit_displacement_to_phase_only = False

        self.setPixelsAndPads()
        self.setGroundTruths()

        self.shared_magnitudes = shared_magnitudes
        self.displacement_tv_reg_const = displacement_tv_reg_const

        with tf.device('/gpu:0'):
            self.coords_t = tf.constant(self.sim.simulations_per_peak[0].nw_coords_stacked, dtype='float32')

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
            if shared_magnitudes:
                size = self.npix_xy + self.sim.params.HKL_list.shape[0] - 1
                self.constraint_fn = None#lambda x: tf.clip_by_value(x, 0, 1e8)
            else:
                size = self.npix_xy * self.sim.params.HKL_list.shape[0]
                self.constraint_fn = None#lambda x: tf.clip_by_value(x, 0., 1.0)

            if magnitudes_init is None:
                magnitudes_init = np.ones(size)
            else:
                if magnitudes_init.size != size:
                    raise ValueError("magnitude initialization supplied is not valid")

            self.magnitudes_v = tf.Variable(magnitudes_init,
                                            constraint = self.constraint_fn,
                                            dtype='float32',
                                            name='magnitudes')
            if ux_uy_init is None:
                ux_uy_init = np.ones(self.npix_xy * 2)
            self.ux_uy_2d_v = tf.Variable(ux_uy_init, dtype='float32', name='ux_uy')

            if phases_init is None:
                self.phases_v = tf.Variable(tf.zeros(self.npix_xy * self.sim.params.HKL_list.shape[0]),
                                            dtype='float32', name='phases')
            else:
                self.phases_v = tf.Variable(phases_init, dtype='float32', name='phases')

            self.unwrapped_phases_v = tf.Variable(self.getUnwrappedPhases(),
                                                  name='unwrapped_phases',
                                                  trainable=False)
            self.setupMinibatch()
        self.optimizers = {}
        self.iteration = tf.constant(0)
        self.objective_loss_per_iter = []

    #@tf.function
    def loss_fn(self, predicted_data_t):
        batch_diff_data_t = tf.reshape(tf.gather(self.diffraction_data_t, self.batch_train_input_v), [-1])
        return 0.5 * tf.reduce_mean((batch_diff_data_t - predicted_data_t)**2)

    def alternating_objective_fn(self):
        return self.objective_fn()

    def convertPhaseToDisplacement(self, phase_flat):
        phases = tf.reshape(phase_flat, (self.displacement_invert_matrix_t.shape[1], -1))
        ux_uy_2d = self.displacement_invert_matrix_t @ phases
        self.ux_uy_2d_v.assign(tf.reshape(ux_uy_2d, [-1]))

    @tf.function
    def optimizersMinimize(self, iters_before_check: tf.Tensor, inner_iters: tf.Tensor):
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

            self.unwrapped_phases_v.assign(self.getUnwrappedPhases())
            self.convertPhaseToDisplacement(self.unwrapped_phases_v)

            phases_from_displacements = tf.reshape(self.get2dPhasesFromDisplacementVars(), [-1])
            phases_wrapped = tf.math.angle(tf.complex(tf.math.cos(phases_from_displacements),
                                                      tf.math.sin(phases_from_displacements)))
            self.phases_v.assign(tf.reshape(phases_wrapped, [-1]))

            objective_loss = self.objective_fn()
            losses_stack.append(objective_loss)

            objective_array = objective_array.write(i, losses_stack)

        return objective_array.stack()


