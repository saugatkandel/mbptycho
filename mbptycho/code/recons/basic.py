import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from mbptycho.code.simulation import Simulation, reloadSimulation
from mbptycho.code.recons.reconstruction_old import ReconstructionT

class ReconstructionRealImagRhosOnlyT(ReconstructionT):
    def __init__(self, simulation: Simulation,
                 batch_size: int = 50,
                 magnitudes_init: np.ndarray = None,
                 phases_init: np.ndarray = None,
                 rhos_init: np.ndarray = None,
                 apply_magnitude_constraint: bool = False,
                 magnitude_constraint_max: float = 1.0):
        """Uses real and imaginary parts of rho as the coordinates of interest. Reconstruct rhos directly
        (and therefore the phase and the magnitudes implicitly).

        Parameters:

        magnitudes_init: np.ndarray[float32]
        Should be of size n_hkl * n_x * n_y. Is set to all ones if not supplied.

        phases_init: np.ndarray[float32]
        Should be of size n_hkl * n_x * n_y.

        rhos_init: np.ndarray[complex64]
        Should be of size n_hkl * n_x * n_y. If rhos_init is supplied, then cannot use magnitudes_init and phases_init.
        """
        self.reconstruction_type = 'phase'
        self.fit_displacement_to_phase_only = False


        self.sim = simulation
        self.batch_size = batch_size

        self.apply_magnitude_constraint = apply_magnitude_constraint
        self.magnitude_constraint_max = magnitude_constraint_max
        if apply_magnitude_constraint:
            raise ValueError("Magnitude constraint not currently implemented.")

        self.setPixelsAndPads()
        self.setGroundTruths()

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

            self.rho_reals_v = tf.Variable(self._getInitializations(rhos_init, magnitudes_init, phases_init),
                                           dtype='float32', name='rho_reals')
            self.setupMinibatch()
            self.optimizers = {}
            self.iteration = tf.constant(0)
            self.epoch = tf.constant(0)
            self.objective_loss_per_iter = []

    def _getInitializations(self, rhos_init, magnitudes_init, phases_init):
        size = self.npix_xy * self.sim.params.HKL_list.shape[0]
        if rhos_init is not None:
            if (magnitudes_init is not None) or (phases_init is not None):
                raise ValueError("Cannot supply magnitudes or phases if rhos are supplied.")
            reals = np.real(rhos_init)
            imag = np.imag(rhos_init)
            inits = np.concatenate([reals.flat, imag.flat], axis=0).astype('float32')
            return inits

        if magnitudes_init is None:
            magnitudes_init = np.ones(size)
        else:
            if magnitudes_init.size != size:
                raise ValueError("magnitude initialization supplied is not valid")

        if phases_init is None:
            phases_init = np.zeros(size)
        else:
            if phases_init.size != size:
                raise ValueError("Phase initialization supplied is not valid.")

        rhos_init = magnitudes_init * np.exp(1j * phases_init)
        reals = np.real(rhos_init)
        imag = np.imag(rhos_init)
        inits = np.concatenate([reals.flat, imag.flat], axis=0)
        return inits

    def get2dRhoFromRealImag(self, rhos_reals_v):
        rhos_reshaped_t = tf.reshape(rhos_reals_v, [2, -1, self.npix_y, self.npix_x])
        rho_2d_cmplx_all_t = tf.complex(rhos_reshaped_t[0], rhos_reshaped_t[1])
        return rho_2d_cmplx_all_t

    def get3dRhoFromRealImag(self, rhos_reals_v):
        rho_2d_cmplx_all_t = self.get2dRhoFromRealImag(rhos_reals_v)
        rho_3d_all_t = tf.ones(self.npix_z, dtype='complex64')[None, None, None, :] * rho_2d_cmplx_all_t[:, :, :, None]
        rho_3d_bordered_all_t = tf.pad(rho_3d_all_t, [[0, 0],
                                                      [self.pady0, self.nyr - self.pady0 - self.npix_y],
                                                      [self.padx0, self.nxr - self.padx0 - self.npix_x],
                                                      [self.padz0, self.nzr - self.padz0 - self.npix_z]])
        return rho_3d_bordered_all_t

    def _batchPredictFromRealImag(self, rhos_reals_v):
        rho_3d_bordered_all_t = self.get3dRhoFromRealImag(rhos_reals_v)
        # Getting the predicted diffraction data. Using a map is faster than a for loop.
        proj_map_fn = lambda indx: self.getProjFtT(rho_3d_bordered_all_t, indx)
        batch_predictions_t = tf.map_fn(proj_map_fn,
                                        self.batch_train_input_v,
                                        dtype=tf.float32)
        out = tf.reshape(batch_predictions_t, [-1])
        return out

    def objective_fn(self):
        return self.data_fit_fn()

    def getNumpyOutputs(self):
        rho_2d_cmplx_all_t = self.get2dRhoFromRealImag(self.rho_reals_v)
        outs = {'rho_2d_phase': rho_2d_cmplx_all_t.numpy()}
        return outs

    def setRhosAdamOptimizer(self, learning_rate):
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['rho'] = {'learning_rate': lr_var,
                                  'optimizer': tf.keras.optimizers.Adam(lr_var),
                                  'var': [self.rho_reals_v]}

    def data_fit_fn(self):
        return self.loss_fn(self._batchPredictFromRealImag(self.rho_reals_v))



