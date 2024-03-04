import numpy as np
from typing import List

class Probes:
    def __init__(
        self, HKL_list: List, probes_h5_file: str, probe_abs_max: float = 1.0
    ):  # probes_matlab_file: str):
        # probes_all = self.loadFullBeamProfiles(probes_matlab_file)
        self._probes_all = self.loadFullBeamProfilesFromH5(probes_h5_file)
        self._probe_abs_max = probe_abs_max
        probes_selected = []
        for H, K, L in HKL_list:
            key_str = f"{H}{K}{L}"
            if key_str in self._probes_all:
                probes_selected.append(self._probes_all[key_str] * probe_abs_max)
            else:
                raise NotImplementedError("not implemented")
        self.probes = probes_selected

    def loadFullBeamProfiles(self, probes_matlab_file):
        from scipy import io as scio

        print("Loading probe from matlab data...")
        probes_all = scio.loadmat(probes_matlab_file)["probes"]
        probes_all = np.array(
            [probes_all[0][i][0] for i in range(probes_all[0].shape[0])]
        )
        print("Loading successful...")
        return probes_all

    def loadFullBeamProfilesFromH5(self, probes_h5_file):
        import h5py

        print("Loading probe from h5py file...")
        probes_all = {}
        with h5py.File(probes_h5_file, "r") as f:
            for k in f.keys():
                probe = f[k][0] + 1j * f[k][1]
                # The tranpose op is for consistency with the probe loaded from .mat files.
                # In the earlier versions of the code, I designed the wave propagation around
                # the probe structure as stored in the .mat files.
                probes_all[k] = probe.T
        print("Loading successfull...")
        return probes_all

    def generateFrom2DProfile(self, pixel_size):
        raise NotImplementedError("Not supported yet")

        from scipy import io as scio

        probe = scio.loadmat("../recon_probe_29902.mat")["prb"]
        probe = np.flipud(probe) / np.max(
            np.abs(probe)
        )  # This gives me a probe structure matching the matlab output
        probe_pixel_size = 10e-3

        x = np.arange(-50, 50) * pixel_size
        Y, X = np.meshgrid(x, x, indexing="ij")

        mask = np.abs(probe) > 0.01  # 25

        vertsprobe1 = np.stack([Y[mask], X[mask], -2 * np.ones(mask.sum())], axis=1)
        vertsprobe2 = np.stack([Y[mask], X[mask], 2 * np.ones(mask.sum())], axis=1)
        vertsprobe = np.concatenate([vertsprobe1, vertsprobe2], axis=0)
