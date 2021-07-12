import mbptycho.code.recons.forward_model as fwdmodel
import mbptycho.code.recons.lossfns as lossfns
#import ptychoSampling.reconstruction.optimization_t

OPTIONS = {"forward models":
               {"displacement_to_phase": fwdmodel.DisplacementPhaseModel,
                "displacement_to_data": fwdmodel.DisplacementFullForwardModel,
                "phase": fwdmodel.PhaseOnlyFullForwardModel,
                "projected": fwdmodel.DisplacementProjectedForwardModel},

           "loss functions":
               {"least_squared":lossfns.LeastSquaredLossT,
                "gaussian": lossfns.LeastSquaredLossT,
                "poisson_log_likelihood": lossfns.PoissonLogLikelihoodLossT,
                "poisson": lossfns.PoissonLogLikelihoodLossT,
                "poisson_surrogate": lossfns.PoissonLogLikelihoodSurrogateLossT,
                "intensity_least_squared": lossfns.IntensityLeastSquaredLossT,
                "counting_model": lossfns.CountingModelLossT}}
 #
 #          "tf_optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.AdamOptimizer,
 #                                      "gradient": ptychoSampling.reconstruction.optimization_t.GradientDescentOptimizer,
 #                                      "momentum": ptychoSampling.reconstruction.optimization_t.MomentumOptimizer
 #                                      }}

           #"optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.getAdamOptimizer,
           #                         "custom": ptychoSampling.reconstruction.optimization_t.getOptimizer}}