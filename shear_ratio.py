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

    def extract_gk_vector(self, cl_dict):
        """
        Extract vector of C_ell^{g_i kappa_j} where mean_z_src > mean_z_lens.
        """
        cl_gk = cl_dict["gk"]
        n_lens, n_src, n_ell = cl_gk.shape

        gk_vector = []
        labels = []

        # Compute mean z for each bin
        mean_z_lens = [np.average(z, weights=nz) for z, nz in self.nz_lens]
        mean_z_src = [np.average(z, weights=nz) for z, nz in self.nz_src]

        for i in range(n_lens):
            for j in range(n_src):
                if mean_z_src[j] > mean_z_lens[i]:
                    gk_vector.append(cl_gk[i, j])
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


    def compute_shear_ratios(self, gk_vector, gk_labels):
        """
        Compute shear ratio data vector R_ij = C^{g_i κ_j} / C^{g_i κ_k}
        for fixed g_i and all j > i, k > j.
    
        Parameters
        ----------
        gk_vector : np.ndarray
            Array of C_l^{g_i kappa_j} values, shape (n_valid, n_ell).
        gk_labels : list of str
            Labels like 'g1k2', 'g1k3', etc., where k > g.
    
        Returns
        -------
        R : np.ndarray
            Shear ratio vector.
        ratio_labels : list of tuple
            Triplet (i, j, k) corresponding to R_ij/R_ik.
        """
        import re
        from itertools import combinations
    
        # Parse (i, j) pairs
        pairs = []
        for label in gk_labels:
            i, j = map(int, re.findall(r'\d+', label))
            pairs.append((i - 1, j - 1))  # zero-based
    
        # Map from (i, j) to index in gk_vector
        pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}
    
        # Build shear ratios
        R = []
        ratio_labels = []
        for i in range(self.nz_lens.__len__()):
            valid_js = [j for (ii, j) in pairs if ii == i and j > i]
            for j, k in combinations(valid_js, 2):
                num = gk_vector[pair_to_index[(i, j)]]
                den = gk_vector[pair_to_index[(i, k)]]
                R.append(num / den)
                ratio_labels.append((i + 1, j + 1, k + 1))  # 1-based for readability
    
        R = np.array(R)  # shape (n_ratios, n_ell)
        return R, ratio_labels

    def sample_gk_vector(self, gk_vector, gk_cov, nsamples=20000, seed=None):
        """
        Draw Monte Carlo realisations of the gk_vector from its covariance.
    
        Parameters
        ----------
        gk_vector : np.ndarray
            Mean vector of C_l^{g_i kappa_j} values (shape: [n, n_ell]).
        gk_cov : np.ndarray
            Covariance matrix of the gk_vector (shape: [n, n]).
        nsamples : int
            Number of Monte Carlo samples to generate.
        seed : int or None
            Random seed for reproducibility.
    
        Returns
        -------
        samples : np.ndarray
            Noisy samples of gk_vector (shape: [nsamples, n, n_ell]).
        """
        rng = np.random.default_rng(seed)
        n, n_ell = gk_vector.shape
        mean = gk_vector[:, 0]  # assuming all ell bins are the same (or just 1 bin)
        samples = rng.multivariate_normal(mean, gk_cov, size=nsamples)
        samples = samples[..., np.newaxis] * np.ones((1, 1, n_ell))  # expand to match original shape
        return samples

    def monte_carlo_shear_ratio_cov(self, gk_vector, gk_cov, gk_labels, nsamples=20000, seed=None):
        """
        Compute the covariance of the shear ratio data vector using MC sampling.
    
        Parameters
        ----------
        gk_vector : np.ndarray
            Mean gk vector, shape (n_valid, n_ell).
        gk_cov : np.ndarray
            Covariance matrix of gk vector, shape (n_valid, n_valid).
        gk_labels : list of str
            Labels for the gk vector elements.
        nsamples : int
            Number of Monte Carlo samples.
        seed : int or None
            Random seed for reproducibility.
    
        Returns
        -------
        R_cov : np.ndarray
            Covariance matrix of the shear-ratio vector, shape (n_ratio, n_ratio).
        R_mean : np.ndarray
            Mean shear-ratio vector, shape (n_ratio,).
        ratio_labels : list of tuple
            List of (i, j, k) labels for each ratio.
        """
        samples = self.sample_gk_vector(gk_vector, gk_cov, nsamples=nsamples, seed=seed)
        n_ell = gk_vector.shape[1]
    
        R_samples = []
        for s in range(nsamples):
            R, labels = self.compute_shear_ratios(samples[s], gk_labels)
            R_samples.append(R[:, 0])  # assuming single ell bin
    
        R_samples = np.array(R_samples)  # (nsamples, n_ratios)
        R_mean = np.mean(R_samples, axis=0)
        R_cov = np.cov(R_samples, rowvar=False)
    
        return R_cov, R_mean, labels


    def compute_shear_ratio_derivatives(self, cl_dict, gk_labels, param_dict, step_frac=0.05):
        """
        Compute numerical derivatives of the shear-ratio data vector w.r.t. cosmological parameters,
        automatically adding delta_z nuisance parameters if toggled during initialization.
        """
        derivs = {}
        gk_vector_fid, _ = self.extract_gk_vector(cl_dict)
        R_fid, ratio_labels = self.compute_shear_ratios(gk_vector_fid, gk_labels)
    
        fid_copy = self.fid_cosmo.copy()
        nz_lens_copy = [ (z.copy(), nz.copy()) for (z, nz) in self.nz_lens ]
        nz_src_copy = [ (z.copy(), nz.copy()) for (z, nz) in self.nz_src ]
    
        # --- cosmological parameters ---
        for p, val in param_dict.items():
            dval = val * step_frac
    
            # Perturb +
            self.fid_cosmo = fid_copy.copy()
            self.fid_cosmo[p] = val + dval
            self._init_cosmology()
            cl_plus = self.compute_cls()
            gk_plus, _ = self.extract_gk_vector(cl_plus)
            R_plus, _ = self.compute_shear_ratios(gk_plus, gk_labels)
    
            # Perturb -
            self.fid_cosmo = fid_copy.copy()
            self.fid_cosmo[p] = val - dval
            self._init_cosmology()
            cl_minus = self.compute_cls()
            gk_minus, _ = self.extract_gk_vector(cl_minus)
            R_minus, _ = self.compute_shear_ratios(gk_minus, gk_labels)
    
            derivs[p] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dval)
    
        # --- delta_z nuisance parameters ---
        dz = 0.002
    
        if self.marginalize_delta_z_lens:
            for i in range(len(self.nz_lens)):
                param_name = f"delta_z_lens_{i}"
    
                self.nz_lens = [(z + (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_lens_copy)]
                self._init_cosmology()
                cl_plus = self.compute_cls()
                gk_plus, _ = self.extract_gk_vector(cl_plus)
                R_plus, _ = self.compute_shear_ratios(gk_plus, gk_labels)
    
                self.nz_lens = [(z - (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_lens_copy)]
                self._init_cosmology()
                cl_minus = self.compute_cls()
                gk_minus, _ = self.extract_gk_vector(cl_minus)
                R_minus, _ = self.compute_shear_ratios(gk_minus, gk_labels)
    
                derivs[param_name] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dz)
    
        if self.marginalize_delta_z_src:
            for i in range(len(self.nz_src)):
                param_name = f"delta_z_src_{i}"
    
                self.nz_src = [(z + (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_src_copy)]
                self._init_cosmology()
                cl_plus = self.compute_cls()
                gk_plus, _ = self.extract_gk_vector(cl_plus)
                R_plus, _ = self.compute_shear_ratios(gk_plus, gk_labels)
    
                self.nz_src = [(z - (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_src_copy)]
                self._init_cosmology()
                cl_minus = self.compute_cls()
                gk_minus, _ = self.extract_gk_vector(cl_minus)
                R_minus, _ = self.compute_shear_ratios(gk_minus, gk_labels)
    
                derivs[param_name] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dz)
        
        # --- multiplicative bias nuisance parameters ---
        dm = 0.005

        if self.marginalize_multiplicative_bias:
            for i in range(len(self.nz_src)):
                param_name = f"m_src_{i}"

                gk_vector_perturb = gk_vector_fid.copy()
                for j, label in enumerate(gk_labels):
                    source_bin = int(label.split('k')[1]) - 1  # parse "g1k2" -> 2 -> src index 1
                    if source_bin == i:
                        gk_vector_perturb[j] = gk_vector_fid[j] * (1 + dm)
                R_plus, _ = self.compute_shear_ratios(gk_vector_perturb, gk_labels)

                gk_vector_perturb = gk_vector_fid.copy()
                for j, label in enumerate(gk_labels):
                    source_bin = int(label.split('k')[1]) - 1
                    if source_bin == i:
                        gk_vector_perturb[j] = gk_vector_fid[j] * (1 - dm)
                R_minus, _ = self.compute_shear_ratios(gk_vector_perturb, gk_labels)

                derivs[param_name] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dm)
    
        # --- restore
        self.fid_cosmo = fid_copy.copy()
        self.nz_lens = nz_lens_copy
        self.nz_src = nz_src_copy
        self._init_cosmology()
    
        return derivs, R_fid[:, 0], ratio_labels

    def compute_shear_ratio_fisher(self, R_cov, derivs):
        """
        Compute Fisher matrix from shear ratio derivatives and covariance.
        Also adds Gaussian priors for delta_z if marginalized.
        """
        from numpy.linalg import inv
    
        inv_cov = inv(R_cov)
        param_list = list(derivs.keys())
        n_params = len(param_list)
    
        fisher = np.zeros((n_params, n_params))
        for i, pi in enumerate(param_list):
            for j, pj in enumerate(param_list):
                dRi = derivs[pi]
                dRj = derivs[pj]
                fisher[i, j] = dRi @ inv_cov @ dRj
    
        # Add Gaussian prior 1/sigma² for delta_z params
        if self.marginalize_delta_z_lens:
            for i in range(len(self.nz_lens)):
                param = f"delta_z_lens_{i}"
                prior_std = self.delta_z_lens_prior_std
                idx = param_list.index(param)
                fisher[idx, idx] += 1.0 / prior_std**2
        
        if self.marginalize_delta_z_src:
            for i in range(len(self.nz_src)):
                param = f"delta_z_src_{i}"
                prior_std = self.delta_z_src_prior_std
                idx = param_list.index(param)
                fisher[idx, idx] += 1.0 / prior_std**2

        if self.marginalize_multiplicative_bias:
            for i in range(len(self.nz_src)):
                param = f"m_src_{i}"
                prior_std = self.multiplicative_bias_prior_std
                idx = param_list.index(param)
                fisher[idx, idx] += 1.0 / prior_std**2
    
        return fisher, param_list

    def plot_fisher_ellipses(self, fisher, param_list, truths=None):
        """
        Plot 2D confidence ellipses (68% and 95%) using Fisher matrix with matplotlib,
        only for cosmological parameters (ignoring nuisance/systematic parameters).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 16,
            "font.size": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })
    
        label_map = {
            "Omega_m": r"$\Omega_m$",
            "Omega_b": r"$\Omega_b$",
            "sigma_8": r"$\sigma_8$",
            "n_s": r"$n_s$",
            "h": r"$h$",
            "w0": r"$w_0$",
            "wa": r"$w_a$"
        }
    
        # Select only cosmological parameters
        cosmo_params = ["Omega_m", "Omega_b", "sigma_8", "n_s", "h", "w0", "wa"]
        selected_params = [p for p in param_list if p in cosmo_params]
    
        if len(selected_params) == 0:
            raise ValueError("No cosmological parameters found to plot.")
    
        idx = [param_list.index(p) for p in selected_params]
        fisher_sel = fisher[np.ix_(idx, idx)]
    
        cov = np.linalg.inv(fisher_sel)
        n_params = len(selected_params)
        mean = np.array([truths[p] if truths else 0 for p in selected_params])
        stds = np.sqrt(np.diag(cov))
    
        fig_width = np.sum(6 * stds)
        fig_height = np.sum(6 * stds)
        fig, axes = plt.subplots(n_params, n_params, figsize=(fig_width, fig_height), squeeze=False)
        plt.subplots_adjust(wspace=0, hspace=0)
    
        chi2_vals = [2.30, 5.99]  # 68% and 95% confidence for 2D Gaussian
    
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
    
                if j > i:
                    ax.axis('off')
                    continue
                elif i == j:
                    sigma = stds[i]
                    x = np.linspace(mean[i] - 3*sigma, mean[i] + 3*sigma, 300)
                    y = np.exp(-0.5*((x - mean[i])/sigma)**2)
                    ax.plot(x, y, color='black')
                    ax.axvline(mean[i], color='black')
                    ax.set_yticks([])
                    ax.set_xlim(mean[i] - 3*sigma, mean[i] + 3*sigma)
                else:
                    sub_cov = cov[[j, i]][:, [j, i]]
                    vals, vecs = np.linalg.eigh(sub_cov)
                    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    
                    for chi2_val, lw in zip(chi2_vals, [1.5, 1]):
                        width, height = 2 * np.sqrt(chi2_val) * np.sqrt(vals)
                        ellipse = Ellipse(xy=(mean[j], mean[i]), width=width, height=height,
                                          angle=angle, edgecolor='black', facecolor='none', linewidth=lw)
                        ax.add_patch(ellipse)
    
                    ax.axvline(mean[j], color='gray')
                    ax.axhline(mean[i], color='gray')
                    ax.set_xlim(mean[j] - 3*stds[j], mean[j] + 3*stds[j])
                    ax.set_ylim(mean[i] - 3*stds[i], mean[i] + 3*stds[i])
    
                if i == n_params - 1:
                    ax.set_xlabel(label_map.get(selected_params[j], f"${selected_params[j]}$"))
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(label_map.get(selected_params[i], f"${selected_params[i]}$"))
                else:
                    ax.set_yticks([])
    
        plt.tight_layout(pad=0)
        plt.show()

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



