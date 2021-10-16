import numpy as np
import tensorflow as tf
from typing import Callable
from skimage.restoration import unwrap_phase
from functools import partial


from mbptycho.code.recons.datalogs import DataLogs
from mbptycho.code.simulation import Simulation
from mbptycho.code.recons.base import  BaseReconstructionT

class DisplacementFromPhaseReconstruction(BaseReconstructionT):
    def __init__(self, simulation, phases_init,
                 log_frequency=1,
                 gpu:str = '/gpu:0'):
        self._gpu = gpu
        self.sim = simulation

        self._model_type = 'displacement_to_phase'
        self.attachForwardModel(self._model_type,
                                shared_magnitudes=None,
                                magnitudes_init=None,
                                phases_init=phases_init,
                                ux_uy_2d_init=None)

        self.iteration = tf.Variable(0, dtype='int32')
        self._iterations_per_epoch = 1
        self._setGroundTruths()

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        self.ux_2d = np.zeros_like(self._ux_true)
        self.uy_2d = np.zeros_like(self._uy_true)

        self.addCustomMetricToDataLog(title="err_ux",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_ux_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true, True),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true, True),
                                      log_epoch_frequency=log_frequency)

        self._finalized = False

    def optimizersMinimize(self):
        pass

    def _addDataLogMetrics(self):
        pass

    def updateOutputs(self):
        ux_2d_t, uy_2d_t = self.fwd_model.getUxUy2d(self.fwd_model.ux_uy_2d_v)
        self.ux_2d = ux_2d_t.numpy()
        self.uy_2d = uy_2d_t.numpy()


    def minimize(self, *args, **kwargs):

        if self._finalized != True:
            self.finalizeInit()

        unwrapped = self.getUnwrappedPhases(self.fwd_model.phases_v)
        new_disps = self.convertPhaseToDisplacement(unwrapped)
        self.fwd_model.ux_uy_2d_v.assign(tf.reshape(new_disps, [-1]))
        self.updateOutputs()

        iter_log_dict = {}
        self.iteration.assign_add(1)
        self._default_log_items["epoch"] = self.epoch

        custom_metrics = self.datalog.getCustomFunctionMetrics(1)
        custom_log_dict = {k:f() for k, f in custom_metrics.items()}

        iter_log_dict.update(self._default_log_items)
        iter_log_dict.update(custom_log_dict)
        self._finalizeIter(iter_log_dict, True, True, 1, 1)

    def saveOutputsAndLog(self, data_path, prefix=''):
        np.savez_compressed(f'{data_path}/{prefix}ux_{self._model_type}.npz', self.ux_2d)
        np.savez_compressed(f'{data_path}/{prefix}uy_{self._model_type}.npz', self.uy_2d)
        self.datalog.dataframe.to_pickle(f'{data_path}/{prefix}df_{self._model_type}.gz')



class DisplacementFullModelReconstruction(BaseReconstructionT):
    def __init__(self, simulation: Simulation,
                 phases_init: np.ndarray = None,
                 magnitudes_init: np.ndarray = None,
                 ux_uy_2d_init: np.ndarray = None,
                 shared_magnitudes: bool = True,
                 loss_type: str = 'gaussian',
                 loss_init_extra_kwargs: dict = None,
                 batch_size: int = 50,
                 background_level: float = 1e-8,
                 n_validation: int = 0,
                 log_frequency=1,
                 unwrap_phase_proj:bool = False):
        self.sim = simulation
        self.batch_size = batch_size

        self.background_level = background_level
        self.unwrap_phase_proj = unwrap_phase_proj

        if phases_init is not None and ux_uy_2d_init is not None:
            raise ValueError("Cannot supply both phases init and ux_uy_2d_init")
        self._model_type = 'displacement_to_data'
        self.attachForwardModel(self._model_type,
                                shared_magnitudes=shared_magnitudes,
                                magnitudes_init=magnitudes_init,
                                phases_init=phases_init,
                                ux_uy_2d_init=ux_uy_2d_init)

        self._splitTrainingValidationData(n_validation, batch_size)
        self._createDataBatches()

        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)


        self.iteration = tf.Variable(0, dtype='int32')
        self._setGroundTruths()

        self.ux_2d = np.zeros_like(self._ux_true)
        self.uy_2d = np.zeros_like(self._uy_true)
        self.rho_2d = np.zeros_like(self._rho_true)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        if n_validation > 0:
            raise ValueError("Not currently supported.")

        self.addCustomMetricToDataLog(title="err_ux",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true, False),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true, False),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_ux_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true, True),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true, True),
                                      log_epoch_frequency=log_frequency)

        # I am using this weird construct because lambda functions misbehave if I don't use indx=indx.
        # See example here: http://math.andrej.com/2009/04/09/pythons-lambda-is-broken/
        # Some "early binding" vs "late binding" problem
        for indx in range(self._rho_true.shape[0]):
            self.addCustomMetricToDataLog(title=f"err_rho{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                               self._rho_true[indx],
                                                                                               False),
                                          log_epoch_frequency=log_frequency)
        for indx in range(self._rho_true.shape[0]):
            self.addCustomMetricToDataLog(title=f"err_rho_film{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                               self._rho_true[indx],
                                                                                               True),
                                          log_epoch_frequency=log_frequency)

        self.optimizers = {}
        self._finalized = False


    def _addDataLogMetrics(self):
        pass

    def updateOutputs(self):
        ux_2d_t, uy_2d_t = self.fwd_model.getUxUy2d(self.fwd_model.ux_uy_2d_v)
        self.ux_2d = ux_2d_t.numpy()
        self.uy_2d = uy_2d_t.numpy()

        magnitudes_t = self.fwd_model.getMagnitudes2d(self.fwd_model.magnitudes_log_v).numpy()
        self.rho_2d = self.fwd_model.get2dRhoFromDisplacements(ux_2d_t, uy_2d_t, magnitudes_t).numpy()


    @tf.function
    def optimizersMinimize(self, iters_before_registration: tf.Tensor):
        objective_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        for i in tf.range(iters_before_registration):
            self._genNewTrainBatch()

            with tf.GradientTape() as gt:
                preds = self._batch_train_preds_fn(ux_uy_2d_v=self.fwd_model.ux_uy_2d_v,
                                                   magnitudes_log_v=self.fwd_model.magnitudes_log_v)
                loss = self._train_loss_fn(preds)
            u_grad, m_grad = gt.gradient(loss, [self.fwd_model.ux_uy_2d_v, self.fwd_model.magnitudes_log_v])

            self.optimizers['ux_uy_2d_v']['optimizer'].apply_gradients(zip([u_grad], [self.fwd_model.ux_uy_2d_v]))
            self.optimizers['magnitudes_log_v']['optimizer'].apply_gradients(zip([m_grad],
                                                                                 [self.fwd_model.magnitudes_log_v]))

            if self.unwrap_phase_proj:
                phases = self.fwd_model.get2dPhasesFromDisplacementVars()
                unwrapped = self.getUnwrappedPhases(tf.reshape(phases, [-1]))
                new_disps = self.convertPhaseToDisplacement(unwrapped)
                self.fwd_model.ux_uy_2d_v.assign(tf.reshape(new_disps, [-1]))

            objective_array = objective_array.write(i, loss)
            self.iteration.assign_add(1)

        return objective_array.stack()

    def minimize(self, max_epochs: int = 500,
                debug_output: bool = True,
                debug_output_epoch_frequency: int = 1):

        if not self._finalized:
            self.finalizeInit()

        print_debug_header = True
        epochs_start = self.epoch
        epochs_this_run = 0

        for i in range(max_epochs):
            losses_array = self.optimizersMinimize(tf.constant(self._iterations_per_epoch)).numpy()
            #losses_array = [1.0] * (self._iterations_per_epoch - 1)
            self.updateOutputs()
            if len(losses_array) > 1:
                for j in range(len(losses_array) - 1):
                    iter_log_dict = {"epoch": self.epoch - 1,
                                     "train_loss": losses_array[j]}
                    self.datalog.logStep(self.iteration.numpy(), iter_log_dict)

            epoch_log_dict = {}
            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = losses_array[-1]

            epoch_log_dict.update(self._default_log_items)
            epochs_this_run = self.epoch - epochs_start
            custom_metrics = self.datalog.getCustomFunctionMetrics(epochs_this_run)
            if len(custom_metrics) > 0:
                custom_log_dict = {k: f() for k, f in custom_metrics.items()}
                epoch_log_dict.update(custom_log_dict)


            print_debug_header = self._finalizeIter(epoch_log_dict, debug_output,
                                                    print_debug_header, epochs_this_run,
                                                    debug_output_epoch_frequency)

    def saveOutputsAndLog(self, data_path, prefix=''):
        suffix = '_shared_mags' if self.fwd_model._shared_magnitudes else '_sep_mags'
        np.savez_compressed(f'{data_path}/{prefix}rho_{self._model_type}{suffix}.npz', self.rho_2d)
        np.savez_compressed(f'{data_path}/{prefix}ux_{self._model_type}{suffix}.npz', self.ux_2d)
        np.savez_compressed(f'{data_path}/{prefix}uy_{self._model_type}{suffix}.npz', self.uy_2d)
        self.datalog.dataframe.to_pickle(f'{data_path}/{prefix}df_{self._model_type}{suffix}.gz')


class PhaseOnlyReconstruction(BaseReconstructionT):
    def __init__(self, simulation: Simulation,
                 phases_init: np.ndarray = None,
                 magnitudes_init: np.ndarray = None,
                 shared_magnitudes: bool = True,
                 loss_type: str = 'gaussian',
                 loss_init_extra_kwargs: dict = None,
                 batch_size: int = 50,
                 background_level: float = 1e-8,
                 n_validation: int = 0,
                 log_frequency: int =1,
                 gpu:str = '/gpu:0'):
        self._gpu = gpu
        self.sim = simulation
        self.batch_size = batch_size

        self.background_level = background_level

        #if phases_init is not None and ux_uy_2d_init is not None:
        #    raise ValueError("Cannot supply both phases init and ux_uy_2d_init")
        self._model_type = 'phase'
        self.attachForwardModel(self._model_type,
                                shared_magnitudes=shared_magnitudes,
                                magnitudes_init=magnitudes_init,
                                phases_init=phases_init)

        self._splitTrainingValidationData(n_validation, batch_size)
        self._createDataBatches()

        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)


        self.iteration = tf.Variable(0, dtype='int32')
        self._setGroundTruths()

        self.rho_2d = np.zeros_like(self._rho_true)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        if n_validation > 0:
            raise ValueError("Not currently supported.")

        for indx in range(self._rho_true.shape[0]):
            self.addCustomMetricToDataLog(title=f"err_rho{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                     self._rho_true[indx],
                                                                                     False),
                                          log_epoch_frequency=log_frequency)

        def temp_fn():
            print(indx)

        for indx in range(self._rho_true.shape[0]):

            self.addCustomMetricToDataLog(title=f"err_rho_film{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                     self._rho_true[indx],
                                                                                     True),
                                          log_epoch_frequency=log_frequency)

        self.optimizers = {}
        self._finalized = False


    def _addDataLogMetrics(self):
        pass

    def updateOutputs(self):
        self.rho_2d = self.fwd_model.get2dRhoFromPhases(self.fwd_model.phases_v,
                                                        self.fwd_model.magnitudes_log_v).numpy()

    @tf.function
    def optimizersMinimize(self, iters_before_registration: tf.Tensor):
        objective_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        for i in tf.range(iters_before_registration):
            self._genNewTrainBatch()

            with tf.GradientTape() as gt:
                preds = self._batch_train_preds_fn(phases_v=self.fwd_model.phases_v,
                                                   magnitudes_log_v=self.fwd_model.magnitudes_log_v)
                loss = self._train_loss_fn(preds)
            u_grad, m_grad = gt.gradient(loss, [self.fwd_model.phases_v, self.fwd_model.magnitudes_log_v])

            self.optimizers['phases_v']['optimizer'].apply_gradients(zip([u_grad], [self.fwd_model.phases_v]))
            self.optimizers['magnitudes_log_v']['optimizer'].apply_gradients(zip([m_grad],
                                                                                 [self.fwd_model.magnitudes_log_v]))
            objective_array = objective_array.write(i, loss)
            self.iteration.assign_add(1)

        return objective_array.stack()

    def minimize(self, max_epochs: int = 500,
                debug_output: bool = True,
                debug_output_epoch_frequency: int = 1):

        if not self._finalized:
            self.finalizeInit()

        print_debug_header = True
        epochs_start = self.epoch
        epochs_this_run = 0

        for i in range(max_epochs):
            losses_array = self.optimizersMinimize(tf.constant(self._iterations_per_epoch)).numpy()
            #losses_array = [1.0] * (self._iterations_per_epoch - 1)
            self.updateOutputs()
            if len(losses_array) > 1:
                for j in range(len(losses_array) - 1):
                    iter_log_dict = {"epoch": self.epoch - 1,
                                     "train_loss": losses_array[j]}
                    self.datalog.logStep(self.iteration.numpy(), iter_log_dict)

            epoch_log_dict = {}
            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = losses_array[-1]

            epoch_log_dict.update(self._default_log_items)
            epochs_this_run = self.epoch - epochs_start
            custom_metrics = self.datalog.getCustomFunctionMetrics(epochs_this_run)
            if len(custom_metrics) > 0:
                custom_log_dict = {k: f() for k, f in custom_metrics.items()}
                epoch_log_dict.update(custom_log_dict)


            print_debug_header = self._finalizeIter(epoch_log_dict, debug_output,
                                                    print_debug_header, epochs_this_run,
                                                    debug_output_epoch_frequency)

    def saveOutputsAndLog(self, data_path, prefix=''):
        suffix = '_shared_mags' if self.fwd_model._shared_magnitudes else '_sep_mags'

        fname_rhos = f'{data_path}/{prefix}rho_{self._model_type}{suffix}.npz'
        print(f"rhos saved in {fname_rhos}")
        np.savez_compressed(fname_rhos, self.rho_2d)

        fname_df = f'{data_path}/{prefix}df_{self._model_type}{suffix}.gz'
        print(f"dataframe saved in {fname_df}")
        self.datalog.dataframe.to_pickle(fname_df)


class DisplacementProjectedReconstruction(BaseReconstructionT):
    def __init__(self, simulation: Simulation,
                 phases_init: np.ndarray = None,
                 magnitudes_init: np.ndarray = None,
                 shared_magnitudes: bool = True,
                 loss_type: str = 'gaussian',
                 loss_init_extra_kwargs: dict = None,
                 batch_size: int = 50,
                 background_level: float = 1e-8,
                 n_validation: int = 0,
                 log_frequency=1,
                 gpu='/gpu:0'):
        self._gpu = gpu
        self.sim = simulation
        self.batch_size = batch_size

        self.background_level = background_level

        #if phases_init is not None and ux_uy_2d_init is not None:
        #    raise ValueError("Cannot supply both phases init and ux_uy_2d_init")
        self._model_type = 'projected'
        self.attachForwardModel(self._model_type,
                                shared_magnitudes=shared_magnitudes,
                                magnitudes_init=magnitudes_init,
                                phases_init=phases_init)

        self._splitTrainingValidationData(n_validation, batch_size)
        self._createDataBatches()

        self.attachLossFunction(loss_type, loss_init_extra_kwargs=loss_init_extra_kwargs)

        self.iteration = tf.Variable(0, dtype='int32')
        self._setGroundTruths()

        self.rho_2d = np.zeros_like(self._rho_true)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        if n_validation > 0:
            raise ValueError("Not currently supported.")

        self.addCustomMetricToDataLog(title="err_ux",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true, False),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true, False),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_ux_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.ux_2d, self._ux_true, True),
                                      log_epoch_frequency=log_frequency)
        self.addCustomMetricToDataLog(title="err_uy_film",
                                      func=lambda: self._getRegistrationErrorDisp(self.uy_2d, self._uy_true, True),
                                      log_epoch_frequency=log_frequency)

        # I am using this weird construct because lambda functions misbehave if I don't use indx=indx.
        # See example here: http://math.andrej.com/2009/04/09/pythons-lambda-is-broken/
        # Some "early binding" vs "late binding" problem
        for indx in range(self._rho_true.shape[0]):
            self.addCustomMetricToDataLog(title=f"err_rho{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                               self._rho_true[indx],
                                                                                               False),
                                          log_epoch_frequency=log_frequency)
        for indx in range(self._rho_true.shape[0]):
            self.addCustomMetricToDataLog(title=f"err_rho_film{indx}",
                                          func=lambda indx=indx: self._getRegistrationErrorRho(self.rho_2d[indx],
                                                                                               self._rho_true[indx],
                                                                                               True),
                                          log_epoch_frequency=log_frequency)

        self.optimizers = {}
        self._finalized = False


    def _addDataLogMetrics(self):
        pass

    def updateOutputs(self):
        self.rho_2d = self.fwd_model.get2dRhoFromPhases(self.fwd_model.phases_v,
                                                        self.fwd_model.magnitudes_log_v).numpy()

        ux_2d_t, uy_2d_t = self.fwd_model.getUxUy2d(self.fwd_model.ux_uy_2d_v)
        self.ux_2d = ux_2d_t.numpy()
        self.uy_2d = uy_2d_t.numpy()

    @tf.function
    def optimizersMinimize(self, iters_before_registration: tf.Tensor):
        objective_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        for i in tf.range(iters_before_registration):
            self._genNewTrainBatch()

            with tf.GradientTape() as gt:
                preds = self._batch_train_preds_fn(phases_v=self.fwd_model.phases_v,
                                                   magnitudes_log_v=self.fwd_model.magnitudes_log_v)
                loss = self._train_loss_fn(preds)
            u_grad, m_grad = gt.gradient(loss, [self.fwd_model.phases_v, self.fwd_model.magnitudes_log_v])

            self.optimizers['phases_v']['optimizer'].apply_gradients(zip([u_grad], [self.fwd_model.phases_v]))
            self.optimizers['magnitudes_log_v']['optimizer'].apply_gradients(zip([m_grad],
                                                                                 [self.fwd_model.magnitudes_log_v]))

            # Unwrapping and projection step
            unwrapped = self.getUnwrappedPhases(self.fwd_model.phases_v)
            new_disps = self.convertPhaseToDisplacement(unwrapped)
            self.fwd_model.ux_uy_2d_v.assign(tf.reshape(new_disps, [-1]))

            ux_2d_t, uy_2d_t = self.fwd_model.getUxUy2d(self.fwd_model.ux_uy_2d_v)
            phases_2d_t = self.fwd_model._get2dPhasesFromDisplacements(ux_2d_t, uy_2d_t)
            self.fwd_model.phases_v.assign(tf.reshape(phases_2d_t, [-1]))

            objective_array = objective_array.write(i, loss)
            self.iteration.assign_add(1)

        return objective_array.stack()

    def minimize(self, max_epochs: int = 500,
                debug_output: bool = True,
                debug_output_epoch_frequency: int = 1):

        if not self._finalized:
            self.finalizeInit()

        print_debug_header = True
        epochs_start = self.epoch
        epochs_this_run = 0

        for i in range(max_epochs):
            losses_array = self.optimizersMinimize(tf.constant(self._iterations_per_epoch)).numpy()
            #losses_array = [1.0] * (self._iterations_per_epoch - 1)
            self.updateOutputs()
            if len(losses_array) > 1:
                for j in range(len(losses_array) - 1):
                    iter_log_dict = {"epoch": self.epoch - 1,
                                     "train_loss": losses_array[j]}
                    self.datalog.logStep(self.iteration.numpy(), iter_log_dict)

            epoch_log_dict = {}
            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = losses_array[-1]

            epoch_log_dict.update(self._default_log_items)
            epochs_this_run = self.epoch - epochs_start
            custom_metrics = self.datalog.getCustomFunctionMetrics(epochs_this_run)
            if len(custom_metrics) > 0:
                custom_log_dict = {k: f() for k, f in custom_metrics.items()}
                epoch_log_dict.update(custom_log_dict)


            print_debug_header = self._finalizeIter(epoch_log_dict, debug_output,
                                                    print_debug_header, epochs_this_run,
                                                    debug_output_epoch_frequency)

    def saveOutputsAndLog(self, data_path, prefix=''):
        suffix = '_shared_mags' if self.fwd_model._shared_magnitudes else '_sep_mags'

        fname_rhos = f'{data_path}/{prefix}rho_{self._model_type}{suffix}.npz'
        print(f"rhos saved in {fname_rhos}")
        np.savez_compressed(fname_rhos, self.rho_2d)
        
        fname_ux = f'{data_path}/{prefix}ux_{self._model_type}{suffix}.npz'
        print(f"ux save in {fname_ux}")
        np.savez_compressed(fname_ux, self.ux_2d)

        fname_uy = f'{data_path}/{prefix}uy_{self._model_type}{suffix}.npz'
        print(f"uy save in {fname_uy}")
        np.savez_compressed(fname_uy, self.uy_2d)

        fname_df = f'{data_path}/{prefix}df_{self._model_type}{suffix}.gz'
        print(f"dataframe saved in {fname_df}")
        self.datalog.dataframe.to_pickle(fname_df)