import numpy as np
import pyccl as ccl
import PySSC
from getdist import MCSamples
import getdist.plots as gdplt

class Forecast:
    def __init__(self,
                 nz_lens,
                 nz_src,
                 ell,
                 area_deg2,
                 n_eff,
                 sigma_eps,
                 bias_lens,
                 fid_cosmo,
                 use_ssc=True,
                 marginalize_delta_z_lens=False,
                 marginalize_delta_z_src=False,
                 delta_z_lens_prior_std=0.002,
                 delta_z_src_prior_std=0.002,
                 marginalize_multiplicative_bias=False,
                 multiplicative_bias_prior_std=0.005):   

        """
        Initialize the forecast object for a 3x2pt analysis.
        """
        self.nz_lens = nz_lens
        self.nz_src = nz_src
        self.ell = np.array(ell)
        self.area_deg2 = area_deg2
        self.n_eff = n_eff
        self.sigma_eps = sigma_eps
        self.bias_lens = np.array(bias_lens)
        self.fid_cosmo = fid_cosmo
        self.use_ssc = use_ssc

        self.marginalize_delta_z_lens = marginalize_delta_z_lens
        self.marginalize_delta_z_src = marginalize_delta_z_src
        self.delta_z_lens_prior_std = delta_z_lens_prior_std
        self.delta_z_src_prior_std = delta_z_src_prior_std
        self.marginalize_multiplicative_bias = marginalize_multiplicative_bias  
        self.multiplicative_bias_prior_std = multiplicative_bias_prior_std      

        self._init_cosmology()


    def _init_cosmology(self):
        """
        Set up pyccl cosmology object.
        """
        self.cosmo = ccl.Cosmology(
            Omega_c=self.fid_cosmo["Omega_m"] - self.fid_cosmo["Omega_b"],
            Omega_b=self.fid_cosmo["Omega_b"],
            h=self.fid_cosmo["h"],
            sigma8=self.fid_cosmo["sigma_8"],
            n_s=self.fid_cosmo["n_s"],
            w0=self.fid_cosmo["w0"],
            wa=self.fid_cosmo["wa"],
            extra_parameters={'camb': {'dark_energy_model': 'ppf'}}
        )

    def plot_nz_bins(self):
        """
        Plot the normalized redshift distributions for lens and source bins.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(self.nz_lens), len(self.nz_src))))

        for i, (z, nz) in enumerate(self.nz_lens):
            axes[0].plot(z, nz, label=f'Lens bin {i+1}', color=colors[i])
        axes[0].set_title("Lens Bins")
        axes[0].set_xlabel("z")
        axes[0].set_ylabel("n(z)")
        axes[0].legend()

        for i, (z, nz) in enumerate(self.nz_src):
            axes[1].plot(z, nz, label=f'Source bin {i+1}', color=colors[i])
        axes[1].set_title("Source Bins")
        axes[1].set_xlabel("z")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def compute_cls(self):
        """
        Compute angular power spectra for gg, gk, and kk.
        """
        n_lens = len(self.nz_lens)
        n_src = len(self.nz_src)
        ell = self.ell

        lens_tracers = []
        for i, (z, nz) in enumerate(self.nz_lens):
            bias = self.bias_lens[i]
            lens_tracers.append(
                ccl.NumberCountsTracer(
                    self.cosmo,
                    has_rsd=False,
                    dndz=(z, nz),
                    bias=(z, bias * np.ones_like(z))
                )
            )

        src_tracers = [
            ccl.WeakLensingTracer(self.cosmo, dndz=(z, nz))
            for (z, nz) in self.nz_src
        ]

        cl_gg = np.zeros((n_lens, n_lens, len(ell)))
        cl_gk = np.zeros((n_lens, n_src, len(ell)))
        cl_kk = np.zeros((n_src, n_src, len(ell)))

        for i in range(n_lens):
            for j in range(i, n_lens):
                cl = ccl.angular_cl(self.cosmo, lens_tracers[i], lens_tracers[j], ell)
                cl_gg[i, j] = cl
                cl_gg[j, i] = cl

        for i in range(n_lens):
            for j in range(n_src):
                cl = ccl.angular_cl(self.cosmo, lens_tracers[i], src_tracers[j], ell)
                cl_gk[i, j] = cl

        for i in range(n_src):
            for j in range(i, n_src):
                cl = ccl.angular_cl(self.cosmo, src_tracers[i], src_tracers[j], ell)
                cl_kk[i, j] = cl
                cl_kk[j, i] = cl

        return {"gg": cl_gg, "gk": cl_gk, "kk": cl_kk}


    def extract_kk_vector(self, cl_dict):
        """
        Extract a flattened data vector from the κκ (weak lensing only) Cls.

        Parameters
        ----------
        cl_dict : dict
            Dictionary of Cls as returned by compute_cls(). Should contain "kk".

        Returns
        -------
        vector : np.ndarray
            Flattened data vector of C_ell^{κκ} for all i <= j.
        labels : list of str
            Labels for each element of the data vector.
        """
        cl_kk = cl_dict["kk"]
        n_src = cl_kk.shape[0]
        ell = self.ell

        vec = []
        labels = []

        for i in range(n_src):
            for j in range(i, n_src):
                label = f"kk_{i}_{j}"
                vec.append(cl_kk[i, j])
                labels.append(label)

        vector = np.concatenate(vec)
        return vector, labels




