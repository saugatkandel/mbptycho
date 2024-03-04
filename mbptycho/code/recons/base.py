import numpy as np
import tensorflow as tf
from typing import Callable
from functools import partial

#import sys
#sys.path.append('../')

from skimage.registration import phase_cross_correlation
from skimage.restoration import unwrap_phase

from mbptycho.code.simulation import Simulation
from mbptycho.code.recons.options import OPTIONS
from mbptycho.code.recons.forward_model import MultiReflectionBraggPtychoFwdModel
from mbptycho.code.recons.datalogs import DataLogs
import abc

class BaseReconstructionT(abc.ABC):

    #@abc.abstractmethod
    def __init__(self, simulation: Simulation,
                 model_type: str,
                 loss_type: str = 'gaussian',
                 loss_init_extra_kwargs: dict = None,
                 batch_size: int = 50,
                 shared_magnitudes: bool = True,
                 magnitudes_init: list = None,
                 phases_init: np.ndarray = None,
                 ux_uy_2d_init: np.ndarray = None,
                 background_level: float = 1e-8,
                 n_validation:int =0):
                 #gpu: str = '/gpu:0'):
        #self._gpu = gpu
        
        self.sim = simulation
        self.batch_size = batch_size
        
        self.background_level = background_level

        self._model_type = model_type
        self.attachForwardModel(model_type,
                                shared_magnitudes=shared_magnitudes,
                                magnitudes_init=magnitudes_init,
                                phases_init=phases_init,
                                ux_uy_2d_init=ux_uy_2d_init)

        self._splitTrainingValidationData(n_validation, batch_size)
        self._createDataBatches()

        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        self.iteration = tf.Variable(0, dtype='int32')
        self._setGroundTruths()

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        if self._n_validation_diffs > 0:
            raise ValueError("Not currently supported.")
            self._validation_log_items = {"validation_loss": None,
                                          "validation_min": None,
                                          "patience": None}
            for key in self._validation_log_items:
                self.datalog.addSimpleMetric(key)

        self._finalized = False

    @property
    def epoch(self):
        return self.iteration.numpy() // self._iterations_per_epoch

    def _splitTrainingValidationData(self, n_validation: int, batch_size: int):
        diffraction_data = [s.diffraction_patterns for s in self.sim.simulations_per_peak]
        s0, s1, s2, s3 = np.shape(diffraction_data)
        self._n_diffs = s0 * s1

        if n_validation % batch_size != 0:
            raise ValueError("'n_validation' should be a multiple of the batch size.")
        self._n_validation_diffs = n_validation
        self._n_train_diffs = self._n_diffs - n_validation

        if batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = self._n_train_diffs


        self._all_indices_shuffled = np.random.permutation(self._n_diffs).astype('int32')
        self._validation_indices = self._all_indices_shuffled[:self._n_validation_diffs]
        self._train_indices = self._all_indices_shuffled[self._n_validation_diffs:]
        self._iterations_per_epoch = self._n_train_diffs // self.batch_size


    def attachForwardModel(self, model_type: str,
                           shared_magnitudes: bool = True,
                           magnitudes_init: list = None,
                           phases_init: np.ndarray = None,
                           ux_uy_2d_init: np.ndarray = None,
                           **extra_kwargs: float):

        models_all =  OPTIONS["forward models"]
        self._checkConfigProperty(models_all, model_type)
        self._attachCustomForwardModel(models_all[model_type],
                                       shared_magnitudes=shared_magnitudes,
                                       magnitudes_init=magnitudes_init,
                                       phases_init=phases_init,
                                       ux_uy_2d_init=ux_uy_2d_init,
                                       **extra_kwargs)


    def _attachCustomForwardModel(self, model: MultiReflectionBraggPtychoFwdModel,
                                  **kwargs):
        self.fwd_model = model(self.sim, **kwargs)

    @staticmethod
    def _checkConfigProperty(options: dict, key_to_check: str):
        if key_to_check not in options:
            e = ValueError(f"{key_to_check} is not currently supported. "
                           + f"Check if {key_to_check} exists as an option among {options} in options.py")
            raise e


    def _setGroundTruths(self):
        # These are the true values. Used for comparison and validation.
        pady0, padx0, nyvar, nxvar = [self.fwd_model._pady0,
                                              self.fwd_model._padx0,
                                              self.fwd_model._npix_y,
                                              self.fwd_model._npix_x]
        nz = self.sim.sample.params.npix_depth // 2
        ux_true = self.sim.sample.Ux_full[..., nz].copy()
        ux_true[~self.sim.sample.obj_mask_w_delta[..., nz]] = 0
        ux_true = ux_true[pady0: pady0 + nyvar, padx0: padx0 + nxvar] / self.sim.sample.params.lattice[0]
        self._ux_true = ux_true #- ux_true.mean()

        uy_true = self.sim.sample.Uy_full[..., nz].copy()
        uy_true[~self.sim.sample.obj_mask_w_delta[..., nz]] = 0
        uy_true = uy_true[pady0: pady0 + nyvar, padx0: padx0 + nxvar] / self.sim.sample.params.lattice[0]
        self._uy_true = uy_true #- uy_true.mean()

        self._rho_true = self.sim.sample.rhos[:, pady0: pady0 + nyvar, padx0: padx0 + nxvar, nz]
        #pady1, padx1 = [self.sim.sample.params.npix_pad_y, self.sim.sample.params.npix_pad_x]
        #self._ux_film_true = (self._ux_true[pady1: -pady1, padx1: -padx1]
        #                      - self._ux_true[pady1: -pady1, padx1: -padx1].mean())
        #self._uy_film_true = (self._uy_true[pady1: -pady1, padx1: -padx1]
        #                      - self._uy_true[pady1: -pady1, padx1: -padx1].mean())
        #
        #self._rho_film_true = self._rho_true[:, pady1: -pady1, padx1: -padx1]


    def getNumpyOutputs(self):
        return self.fwd_model.getNumpyOutputs()

    def _getBatchedDataIterate(self, data_tensor: tf.Tensor):
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = dataset.shuffle(data_tensor.get_shape()[0])
        dataset = dataset.repeat()

        dataset_batch = dataset.batch(self.batch_size, drop_remainder=True)
        dataset_batch = dataset_batch.prefetch(tf.data.AUTOTUNE)
        #dataset_batch.apply(tf.data.Dataset.prefetch(tf.data.AUTOTUNE))#/experimental.prefetch_to_device(self._gpu, 2))

        iterator = iter(dataset_batch)
        return iterator

    def _createDataBatches(self):

        # The diffraction patterns.
        # I am including all the diffraction patterns (for all the bragg peaks) as one giant dataset.
        # During the minibatch optimization process, I select the specified number (minibatch size)  of diffraction
        # patterns randomly.


        measured_magnitudes_t = tf.constant(np.array([s.diffraction_patterns for s in self.sim.simulations_per_peak]),
                                            dtype='float32')**0.5
        self.measured_magnitudes_t = tf.reshape(measured_magnitudes_t,
                                                [-1, *self.sim.simulations_per_peak[0].diffraction_patterns[0].shape])

        # the total number of diffraction patterns is len(sm.ptycho_scan_positions) * len(sm.params.HKL_list)
        # We select a minibatch from there.
        self._train_iterator = self._getBatchedDataIterate(tf.constant(self._train_indices))
        if self._n_validation_diffs > 0:
            self._validation_iterator = self._getBatchedDataIterate(tf.constant(self._validation_indices))

        # I am using an extra variable here to ensure that the minibatch is only updated exactly when
        # I want.
        # This variable is not an optimization variable.
        #with tf.device(self._gpu):
        self._batch_train_input_v = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32), trainable=False)
        if self._n_validation_diffs > 0:
            self._batch_validation_input_v = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int32), trainable=False)

    def _genNewTrainBatch(self):
        train_batch_indices = self._batch_train_input_v.assign(self._train_iterator.next())
        # Selecting the measured diffraction data for the minibatch
        return train_batch_indices

    def _genNewValidationbatch(self):
        validation_batch_indices = self._batch_validation_input_v.assign(self._validation_iterator.next())
        # Selecting the measured diffraction data for the minibatch
        return validation_batch_indices

    #@abc.abstractmethod
    def _attachModelPredictions(self, map_preds_fn: Callable=None, map_data_fn: Callable=None):
        if map_preds_fn is None:
            map_preds_fn = lambda x: x
        if map_data_fn is None:
            map_data_fn = lambda x: x

        preds_fn = lambda *args, **kwargs: map_preds_fn(self.fwd_model.predict(*args, **kwargs))

        self._batch_train_preds_fn = partial(preds_fn, self._batch_train_input_v)

        self._batch_train_data_fn = lambda: map_data_fn(tf.reshape(tf.gather(self.measured_magnitudes_t,
                                                                             self._batch_train_input_v), [-1]))
        if self._n_validation_diffs > 0:
            self._validation_preds_fn = partial(preds_fn, self._batch_validation_input_v)
            self._batch_validation_data_fn = lambda: map_data_fn(tf.reshape(tf.gather(self.measured_magnitudes_t,
                                                                             self._batch_validation_input_v), [-1]))


    def attachLossFunction(self, loss_type: str, loss_init_extra_kwargs: dict=None):
        losses_all = OPTIONS["loss functions"]

        self._checkConfigProperty(losses_all, loss_type)
        loss_method = losses_all[loss_type]

        if loss_init_extra_kwargs is None:
            loss_init_extra_kwargs = {}
        if 'background_level' in loss_init_extra_kwargs and self.background_level is not None:
            raise ValueError("Cannot supply background level in loss argument if " +
                             "'background_level' in main class is not None.")
        loss_init_args= {'background_level': self.background_level}
        loss_init_args.update(loss_init_extra_kwargs)


        self._checkAttr("fwd_model", "loss functions")

        self._loss_method = loss_method(**loss_init_args)
        self._attachModelPredictions(self._loss_method.map_preds_fn, self._loss_method.map_data_fn)

        self._train_loss_fn = lambda p: self._loss_method.loss_fn(p, self._batch_train_data_fn())

        if hasattr(self._loss_method, "hessian_fn"):
            self._train_hessian_fn = lambda p: self._loss_method.hessian_fn(p, self._batch_train_data_fn())
        else:
            self._train_hessian_fn = None

        self._validation_loss_fn = lambda p: self._loss_method.loss_fn(p, self._batch_validation_data_t)

    def addRegularizer(self, regularized_tensor, regularization_constant, name: str):
        raise AttributeError("Not currently supported.")
        if not hasattr(self, "regularizers"):
            self.regularizers = {}
        reg_const = tf.Variable(regularization_constant, dtype='float32')
        self.regularizers[name] = {'reg_const': reg_const,
                                   'tensor': regularized_tensor}

    def _checkAttr(self, attr_to_check, attr_this):
        if not hasattr(self, attr_to_check):
            e = AttributeError(f"First attach a {attr_to_check} before attaching {attr_this}.")
            #logger.error(e)
            raise e

    def setDisplacementAdamOptimizerOld(self, learning_rate):
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        if not hasattr(self.fwd_model, "ux_uy_2d_v"):
            raise ValueError(f"Cannot optimize for the displacement variable in the '{self._model_type}' forward model")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['ux_uy_2d_v'] = {'learning_rate': lr_var,
                                           'optimizer': tf.keras.optimizers.Adam(lr_var),
                                           'var': self.fwd_model.ux_uy_2d_v}
    def setDisplacementAdamOptimizer(self, learning_rate):
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        if not hasattr(self.fwd_model, "ux_uy_2d_v"):
            raise ValueError(f"Cannot optimize for the displacement variable in the '{self._model_type}' forward model")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['ux_uy_2d_v'] = {'learning_rate': lr_var,
                                           'optimizer': 
                                         tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(lr_var)),
                                           'var': self.fwd_model.ux_uy_2d_v}

    def setMagnitudeLogAdamOptimizerOld(self, learning_rate):
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        if not hasattr(self.fwd_model, "magnitudes_log_v"):
            raise ValueError(f"Cannot optimize for the magnitude variable in the '{self._model_type}' forward model")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['magnitudes_log_v'] = {'learning_rate': lr_var,
                                        'optimizer': tf.keras.optimizers.Adam(lr_var),
                                        'var': self.fwd_model.magnitudes_log_v}
    def setMagnitudeLogAdamOptimizer(self, learning_rate):
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        if not hasattr(self.fwd_model, "magnitudes_log_v"):
            raise ValueError(f"Cannot optimize for the magnitude variable in the '{self._model_type}' forward model")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['magnitudes_log_v'] = {'learning_rate': lr_var,
                                        'optimizer':
                                            tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(lr_var)),
                                        'var': self.fwd_model.magnitudes_log_v}

    def setPhaseAdamOptimizer(self, learning_rate):
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        if not hasattr(self.fwd_model, "phases_v"):
            raise ValueError(f"Cannot optimize for the phase variable in the '{self._model_type}' forward model")
        lr_var = tf.Variable(learning_rate, dtype='float32')
        self.optimizers['phases_v'] = {'learning_rate': lr_var,
                                    'optimizer': 
                                       tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(lr_var)),
                                       #tf.keras.optimizers.Adam(lr_var),
                                    'var': self.fwd_model.phases_v}

    def convertPhaseToDisplacement(self, phase_flat):
        phases = tf.reshape(phase_flat, (self.fwd_model._displacement_invert_matrix_t.shape[1], -1))
        ux_uy_2d = self.fwd_model._displacement_invert_matrix_t @ phases
        return ux_uy_2d
        #self.ux_uy_2d_v.assign(tf.reshape(ux_uy_2d, [-1]))

    def getUnwrappedPhases(self, phase_flat):

        def _unwrap_phases(phases):
            phases_reshaped = np.reshape(phases, [-1, self.fwd_model._npix_y, self.fwd_model._npix_x])
            unwrapped = []
            for p in phases_reshaped:
                p_new = unwrap_phase(p)
                unwrapped.append(p_new)
            return np.array(unwrapped, 'float32').reshape(-1)

        phases_unwrapped = tf.py_function(_unwrap_phases, inp=[phase_flat],
                                          Tout=[tf.float32])

        return tf.reshape(phases_unwrapped, [-1])

    def addCustomMetricToDataLog(self, title: str,
                                 func: Callable,
                                 log_epoch_frequency: int = 1,
                                 registration_ground_truth: np.ndarray=None,
                                 registration: bool = True,
                                 normalized_lse: bool = False):
        """Registration metric type only applies if registration ground truth is not none."""
        if registration_ground_truth is None:
            self.datalog.addCustomFunctionMetric(title=title, func=func, log_epoch_frequency=log_epoch_frequency)
        else:
            if registration and normalized_lse:
                e =  ValueError("Only one of 'registration' or 'normalized lse' should be true.")
                #logger.error(e)
                raise e

            self.datalog.addCustomFunctionMetric(title=title,
                                               func=func,
                                               registration=registration,
                                               normalized_lse=normalized_lse,
                                               log_epoch_frequency=log_epoch_frequency,
                                               true=registration_ground_truth)

    @abc.abstractmethod
    def optimizersMinimize(self, iters_before_registration: tf.Tensor):
        pass

    def _get2DFilmOnly(self, input_2d):
        pady1, padx1 = [self.sim.sample.params.npix_delta_y,
                        self.sim.sample.params.npix_delta_x]
        # this gives errors when pady1 or padx1 is 0.
        #output_2d = input_2d[pady1: -pady1, padx1: -padx1]
        output_2d = input_2d[pady1: input_2d.shape[0] - pady1, padx1: input_2d.shape[1] - padx1]
        return output_2d

    def _getRegistrationErrors(self, test, true, film_only=False, subtract_mean=False):
        if film_only:
            test = self._get2DFilmOnly(test)
            true = self._get2DFilmOnly(true)

        if subtract_mean:
            test = test - test.mean()
            true = true - true.mean()
        
        roll, err, phase = phase_cross_correlation(true, test, upsample_factor=10)
        roll, err, phase = phase_cross_correlation(true, test * np.exp(1j * phase), upsample_factor=10)
        return err

    def _getRegistrationErrorDisp(self, u_test, u_true, film_only=False):
        return self._getRegistrationErrors(u_test, u_true, film_only=film_only, subtract_mean=True)

    def _getRegistrationErrorRho(self, rho_test, rho_true, film_only=False):
        return self._getRegistrationErrors(rho_test, rho_true, film_only=film_only, subtract_mean=False)


    def _printDebugOutput(self, debug_output_epoch_frequency, epoch, print_debug_header):
        if not epoch % debug_output_epoch_frequency == 0:
            return print_debug_header

        self.datalog.printDebugOutput(print_debug_header)
        return False

    def finalizeInit(self):
        print("Initializing the datalog...")
        self.datalog.finalize()
        self._finalized = True

    def _finalizeIter(self, iter_log_dict,
                      debug_output_this_iter,
                      print_debug_header,
                      epochs_this_run,
                      debug_output_epoch_frequency):
        self.datalog.logStep(self.iteration.numpy(), iter_log_dict)
        out = print_debug_header
        if debug_output_this_iter:
            out = self._printDebugOutput(debug_output_epoch_frequency,
                                         epochs_this_run,
                                         print_debug_header)
        return out

    @abc.abstractmethod
    def minimize(self, *args, **kwarsg):
        pass


