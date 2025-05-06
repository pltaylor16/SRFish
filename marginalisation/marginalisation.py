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
                 amplitude_prior_std=None,
                 marginalize_delta_z_lens=False,
                 marginalize_delta_z_src=False,
                 delta_z_lens_prior_std=0.002,
                 delta_z_src_prior_std=0.002,
                 marginalize_multiplicative_bias=False,
                 multiplicative_bias_prior_std=0.005):
        """
        Initialize the forecast object.
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

        # Gaussian prior std on linear biases (mean 0), one per lens bin
        self.amplitude_prior_std = amplitude_prior_std

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

    # --- Update extract_gk_vector method ---
    def extract_gk_vector(self, cl_dict):
        """
        Extract vector of C_ell^{g_i kappa_j} where mean_z_src > mean_z_lens.
        """
        cl_gk = cl_dict["gk"]
        n_lens, n_src, n_ell = cl_gk.shape

        gk_vector = []
        labels = []

        mean_z_lens = [np.average(z, weights=nz) for z, nz in self.nz_lens]
        mean_z_src = [np.average(z, weights=nz) for z, nz in self.nz_src]

        for i in range(n_lens):
            for j in range(n_src):
                if mean_z_src[j] > mean_z_lens[i]:
                    val = cl_gk[i, j]
                    gk_vector.append(val)
                    labels.append(f"g{i+1}k{j+1}")

        gk_vector = np.array(gk_vector)
        return gk_vector, labels

    def compute_gk_covariance_matrix_gauss(self, cl_dict, ell, delta_ell):
        """
        Compute the Gaussian covariance matrix for gk_vector, matching the new (mean(z_src) > mean(z_lens)) logic.
        """
        cl_gg = cl_dict["gg"]
        cl_gk = cl_dict["gk"]
        cl_kk = cl_dict["kk"]
        n_lens = len(self.nz_lens)
        n_src = len(self.nz_src)

        # Compute mean redshifts
        lens_means = [np.average(z, weights=nz) for (z, nz) in self.nz_lens]
        src_means  = [np.average(z, weights=nz) for (z, nz) in self.nz_src]

        # Select only valid (lens, source) pairs
        indices = []
        labels = []
        for i in range(n_lens):
            for j in range(n_src):
                if src_means[j] > lens_means[i]:
                    indices.append((i, j))
                    labels.append(f"g{i+1}k{j+1}")

        N = len(indices)

        # Survey information
        steradian_conversion = 3600 * (180 / np.pi)**2
        nbar_lens = self.n_eff * steradian_conversion / n_lens
        nbar_src = self.n_eff * steradian_conversion / n_src
        sigma_e_sq = self.sigma_eps ** 2

        f_sky = self.area_deg2 / 41253.
        prefac = 1 / ((2 * ell + 1) * f_sky * delta_ell)

        cov = np.zeros((N, N))

        for a, (i, j) in enumerate(indices):
            for b, (k, l) in enumerate(indices):
                C_gg_val = cl_gg[i, k, 0]
                if i == k:
                    C_gg_val += 1.0 / nbar_lens

                C_kk_val = cl_kk[j, l, 0]
                if j == l:
                    C_kk_val += sigma_e_sq / nbar_src

                C_gl_val = cl_gk[i, l, 0]
                C_kj_val = cl_gk[k, j, 0]

                cov[a, b] = prefac * (C_gg_val * C_kk_val + C_gl_val * C_kj_val)

        return cov, labels


    def compute_ssc_covariance_pyssc(self, cl_dict, ell_eff, mask_fsky=1.0):
        """
        Compute SSC covariance matrix for gk vector using PySSC,
        keeping only pairs with mean_z_src > mean_z_lens.
        """
        import PySSC

        z_arr = self.nz_src[0][0]
        n_lens = len(self.nz_lens)
        n_src = len(self.nz_src)
        n_total = n_lens + n_src

        kernels = np.zeros((n_total, z_arr.size))
        for i, (z, nz) in enumerate(self.nz_lens):
            kernels[i, :] = nz / np.trapz(nz, z) * self.bias_lens[i]

        for j, (z, nz) in enumerate(self.nz_src):
            a = 1.0 / (1.0 + z)
            chi = self.cosmo.comoving_radial_distance(a)

            nz_norm = nz / np.trapz(nz, z)
            chi_of_z = self.cosmo.comoving_radial_distance(a)

            integrand = np.zeros_like(z)
            for idx, z_i in enumerate(z):
                chi_i = chi[idx]
                mask = chi > chi_i
                integrand[idx] = np.trapz(
                    nz_norm[mask] * (chi[mask] - chi_i) / chi[mask], z[mask]
                )

            prefactor = (3/2) * self.fid_cosmo['Omega_m'] * (self.fid_cosmo['h']**2) * (ccl.physical_constants.CLIGHT/1000.)**2
            kernels[n_lens + j, :] = prefactor * chi * (1 + z) * integrand

        cosmo_params = {
            'h': self.fid_cosmo['h'],
            'Omega_m': self.fid_cosmo['Omega_m'],
            'Omega_b': self.fid_cosmo['Omega_b'],
            'sigma8': self.fid_cosmo['sigma_8'],
            'n_s': self.fid_cosmo['n_s'],
            'P_k_max_h/Mpc': 20.0,
            'z_max_pk': 5.0,
            'output': 'mPk',
        }

        Sijkl = PySSC.Sijkl(z_arr, kernels, cosmo_params=cosmo_params)

        gk_vector, labels = self.extract_gk_vector(cl_dict)  # uses updated logic
        N = len(labels)

        # Precompute mean redshifts
        mean_z_lens = [np.average(z, weights=nz) for z, nz in self.nz_lens]
        mean_z_src = [np.average(z, weights=nz) for z, nz in self.nz_src]

        cov_ssc = np.zeros((N, N))

        def _decode(lbl):
            gi, kj = lbl[1:].split('k')
            return int(gi) - 1, int(kj) - 1

        for a in range(N):
            i, j = _decode(labels[a])
            for b in range(N):
                m, n = _decode(labels[b])

                if (mean_z_src[j] > mean_z_lens[i]) and (mean_z_src[n] > mean_z_lens[m]):
                    R_C_a = (self.bias_lens[i]) * 3. * cl_dict["gk"][i, j, 0]
                    R_C_b = (self.bias_lens[m]) * 3. * cl_dict["gk"][m, n, 0]
                    cov_ssc[a, b] = R_C_a * R_C_b * Sijkl[i, j, m, n]
                else:
                    cov_ssc[a, b] = 0.0

        return cov_ssc, labels

    def compute_gk_covariance_matrix(self, cl_dict, ell, delta_ell):
        """
        Compute the full covariance matrix (Gaussian + SSC if use_ssc=True).
        """
        cov_gauss, labels = self.compute_gk_covariance_matrix_gauss(cl_dict, ell, delta_ell)
        if self.use_ssc:
            cov_ssc, _ = self.compute_ssc_covariance_pyssc(cl_dict, ell_eff=ell)
            return cov_gauss + cov_ssc, labels
        else:
            return cov_gauss, labels


    def compute_derivatives_gk(self, cl_fid, param_names):
        """
        Compute derivatives of gk_vector w.r.t. cosmological parameters and linear biases,
        using 5% finite differences. Linear biases are labeled as 'bias_i'.
        """
        derivs = []
        gk_fid, labels = self.extract_gk_vector(cl_fid)

        # --- Cosmological parameter derivatives ---
        for pname in param_names:
            val = self.fid_cosmo[pname]
            step = 0.05 * abs(val)
            p_up = self.fid_cosmo.copy(); p_up[pname] += step
            p_down = self.fid_cosmo.copy(); p_down[pname] -= step

            up_forecast = Forecast(
                nz_lens=self.nz_lens,
                nz_src=self.nz_src,
                ell=self.ell,
                area_deg2=self.area_deg2,
                n_eff=self.n_eff,
                sigma_eps=self.sigma_eps,
                bias_lens=self.bias_lens,
                fid_cosmo=p_up,
                use_ssc=self.use_ssc,
                amplitude_prior_std=self.amplitude_prior_std
            )
            down_forecast = Forecast(
                nz_lens=self.nz_lens,
                nz_src=self.nz_src,
                ell=self.ell,
                area_deg2=self.area_deg2,
                n_eff=self.n_eff,
                sigma_eps=self.sigma_eps,
                bias_lens=self.bias_lens,
                fid_cosmo=p_down,
                use_ssc=self.use_ssc,
                amplitude_prior_std=self.amplitude_prior_std
            )

            gk_up, _ = up_forecast.extract_gk_vector(up_forecast.compute_cls())
            gk_down, _ = down_forecast.extract_gk_vector(down_forecast.compute_cls())
            deriv = (gk_up - gk_down) / (2 * step)
            derivs.append(deriv.flatten())

        full_param_names = list(param_names)

        # --- Bias parameter derivatives ---
        for i in range(len(self.bias_lens)):
            bias_up = self.bias_lens.copy()
            bias_down = self.bias_lens.copy()
            step = 0.05 * abs(bias_up[i])
            bias_up[i] += step
            bias_down[i] -= step

            up_forecast = Forecast(
                nz_lens=self.nz_lens,
                nz_src=self.nz_src,
                ell=self.ell,
                area_deg2=self.area_deg2,
                n_eff=self.n_eff,
                sigma_eps=self.sigma_eps,
                bias_lens=bias_up,
                fid_cosmo=self.fid_cosmo,
                use_ssc=self.use_ssc,
                amplitude_prior_std=self.amplitude_prior_std
            )
            down_forecast = Forecast(
                nz_lens=self.nz_lens,
                nz_src=self.nz_src,
                ell=self.ell,
                area_deg2=self.area_deg2,
                n_eff=self.n_eff,
                sigma_eps=self.sigma_eps,
                bias_lens=bias_down,
                fid_cosmo=self.fid_cosmo,
                use_ssc=self.use_ssc,
                amplitude_prior_std=self.amplitude_prior_std
            )

            gk_up, _ = up_forecast.extract_gk_vector(up_forecast.compute_cls())
            gk_down, _ = down_forecast.extract_gk_vector(down_forecast.compute_cls())
            deriv = (gk_up - gk_down) / (2 * step)
            derivs.append(deriv.flatten())
            full_param_names.append(f"bias_{i}")

        return np.array(derivs), full_param_names

    def compute_fisher_gk(self, cl_fid, cov, param_names):
        """
        Compute Fisher matrix for gk_vector. Adds Gaussian priors if amplitude_prior_std is specified.
        """
        derivs, full_param_names = self.compute_derivatives_gk(cl_fid, param_names)
        cov_inv = np.linalg.inv(cov)

        n_params = len(full_param_names)
        F = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(n_params):
                F[i, j] = derivs[i] @ cov_inv @ derivs[j]

        # Add Gaussian priors on bias parameters (assumed to be named "bias_i")
        if self.amplitude_prior_std is not None:
            for i, pname in enumerate(full_param_names):
                if pname.startswith("bias_"):
                    idx = int(pname.split("_")[1])
                    prior_std = self.amplitude_prior_std[idx]
                    F[i, i] += 1.0 / (prior_std ** 2)

        return full_param_names, F


    def draw_fisher_samples(self, fisher, param_list, truths, n_samples=10000):
        from numpy.linalg import inv
    
        cov = inv(fisher)
        rng = np.random.default_rng()
    
        samples = rng.multivariate_normal(mean=np.zeros(len(param_list)), cov=cov, size=n_samples)
    
        # Shift samples by truth values
        mean_vals = np.array([truths.get(p, 0.0) for p in param_list])  # 0 for systematics like delta_z, m_src
        shifted_samples = samples + mean_vals
    
        return shifted_samples, param_list

    def plot_fisher_samples_with_getdist(self, samples, param_names, truths):
        import numpy as np
        from getdist import MCSamples
        import getdist.plots as gdplt
        import matplotlib.pyplot as plt
    
        # Only cosmological parameters for plotting
        cosmo_params = ["Omega_m", "Omega_b", "sigma_8", "n_s", "h", "w0", "wa"]
    
        # Filter param_names to cosmological ones
        param_names_cosmo = [p for p in param_names if p in cosmo_params]
        idx_cosmo = [param_names.index(p) for p in param_names_cosmo]
    
        # Select only the cosmological samples
        samples_cosmo = samples[:, idx_cosmo]
    
        # Prettier labels
        label_map = {
            "Omega_m": r"\Omega_m",
            "Omega_b": r"\Omega_b",
            "sigma_8": r"\sigma_8",
            "n_s": r"n_s",
            "h": r"h",
            "w0": r"w_0",
            "wa": r"w_a",
        }
        labels = [label_map[p] for p in param_names_cosmo]
    
        gdsamples = MCSamples(samples=samples_cosmo, names=param_names_cosmo, labels=labels)
    
        # Turn on LaTeX
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 16,
            "font.size": 14,
            "legend.fontsize": 12,
        })
    
        # Plot
        g = gdplt.get_subplot_plotter()
        g.triangle_plot(gdsamples, filled=True)













