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
        Extract a vector of C_ell^{κ_i κ_j} for i <= j.

        Returns
        -------
        kk_vector : np.ndarray
            Array of shape (n_valid_pairs, n_ell)
        labels : list of str
            Labels for each κκ pair, like 'kk_0_2'
        """
        cl_kk = cl_dict["kk"]
        n_src, _, n_ell = cl_kk.shape

        vec = []
        labels = []

        for i in range(n_src):
            for j in range(i, n_src):
                vec.append(cl_kk[i, j])  # shape (n_ell,)
                labels.append(f"kk_{i}_{j}")

        kk_vector = np.stack(vec, axis=0)  # shape (n_valid_pairs, n_ell)
        return kk_vector, labels


    def compute_kk_covariance_matrix_gauss(self, cl_dict, ell, delta_ell):
        """
        Compute the Gaussian covariance matrix for the κκ (weak lensing) angular power spectra.
        
        Parameters
        ----------
        cl_dict : dict
            Dictionary of Cls as returned by compute_cls(), must include "kk".
        ell : float
            Effective multipole ℓ at which the Cls are evaluated.
        delta_ell : int
            Width of the ℓ band over which Cls are averaged.
        
        Returns
        -------
        cov : ndarray
            Gaussian covariance matrix for the κκ data vector.
        labels : list of str
            Labels for each Cl entry (kk_i_j) in the vector.
        """
        cl_kk = cl_dict["kk"]
        n_src = len(self.nz_src)

        # Generate valid index pairs for i <= j
        indices = []
        labels = []
        for i in range(n_src):
            for j in range(i, n_src):
                indices.append((i, j))
                labels.append(f"kk_{i}_{j}")
        
        N = len(indices)

        # Survey properties
        f_sky = self.area_deg2 / 41253.
        steradian_conversion = 3600 * (180 / np.pi)**2
        nbar_src = self.n_eff * steradian_conversion / n_src
        sigma_e_sq = self.sigma_eps ** 2
        prefac = 1 / ((2 * ell + 1) * f_sky * delta_ell)

        cov = np.zeros((N, N))

        for a, (i, j) in enumerate(indices):
            for b, (k, l) in enumerate(indices):
                # Wick contractions
                C_ik = cl_kk[i, k, 0] + (sigma_e_sq / nbar_src if i == k else 0.0)
                C_jl = cl_kk[j, l, 0] + (sigma_e_sq / nbar_src if j == l else 0.0)
                C_il = cl_kk[i, l, 0] + (sigma_e_sq / nbar_src if i == l else 0.0)
                C_jk = cl_kk[j, k, 0] + (sigma_e_sq / nbar_src if j == k else 0.0)

                cov[a, b] = prefac * (C_ik * C_jl + C_il * C_jk)

        return cov, labels

    def compute_ssc_covariance_kk_pyssc(self, cl_dict, ell_eff, mask_fsky=1.0):
        """
        Compute SSC covariance matrix for κκ vector using PySSC.
        Only includes i <= j and k <= l terms.
        
        Parameters
        ----------
        cl_dict : dict
            Dictionary of Cls as returned by compute_cls().
        ell_eff : float
            Effective multipole at which Cls are evaluated.
        mask_fsky : float, optional
            Sky fraction (default is 1.0 for full area_deg2 coverage).
        
        Returns
        -------
        cov_ssc : ndarray
            SSC covariance matrix for κκ data vector.
        labels : list of str
            Labels matching the vector.
        """
        import PySSC

        z_arr = self.nz_src[0][0]
        n_src = len(self.nz_src)

        # --- Build lensing kernels for source bins ---
        kernels = np.zeros((n_src, z_arr.size))

        for j, (z, nz) in enumerate(self.nz_src):
            a = 1.0 / (1.0 + z)
            chi = self.cosmo.comoving_radial_distance(a)
            nz_norm = nz / np.trapz(nz, z)

            integrand = np.zeros_like(z)
            for idx, z_i in enumerate(z):
                chi_i = chi[idx]
                mask = chi > chi_i
                integrand[idx] = np.trapz(
                    nz_norm[mask] * (chi[mask] - chi_i) / chi[mask], z[mask]
                )

            prefactor = (3/2) * self.fid_cosmo['Omega_m'] * (self.fid_cosmo['h']**2) * (ccl.physical_constants.CLIGHT/1000.)**2
            kernels[j, :] = prefactor * chi * (1 + z) * integrand

        # --- PySSC setup ---
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

        # Extract vector and labels
        kk_vector, labels = self.extract_kk_vector(cl_dict)
        N = len(labels)

        cov_ssc = np.zeros((N, N))

        def _decode(lbl):
            i, j = map(int, lbl[3:].split('_'))
            return i, j

        for a in range(N):
            i, j = _decode(labels[a])
            for b in range(N):
                k, l = _decode(labels[b])

                R_C_a = 3. * cl_dict["kk"][i, j, 0]
                R_C_b = 3. * cl_dict["kk"][k, l, 0]

                cov_ssc[a, b] = R_C_a * R_C_b * Sijkl[i, j, k, l]

        return cov_ssc, labels


    def compute_kk_covariance_matrix(self, cl_dict, ell, delta_ell):
        """
        Compute the full covariance matrix (Gaussian + SSC if use_ssc=True)
        for the κκ angular power spectrum data vector.
        
        Parameters
        ----------
        cl_dict : dict
            Dictionary of Cls as returned by compute_cls().
        ell : float
            Effective multipole.
        delta_ell : int
            Width of the ℓ band.

        Returns
        -------
        cov : ndarray
            Covariance matrix for the κκ data vector.
        labels : list of str
            Labels for each element in the data vector.
        """
        cov_gauss, labels = self.compute_kk_covariance_matrix_gauss(cl_dict, ell, delta_ell)
        
        if self.use_ssc:
            cov_ssc, _ = self.compute_ssc_covariance_kk_pyssc(cl_dict, ell_eff=ell)
            return cov_gauss + cov_ssc, labels
        else:
            return cov_gauss, labels

    def compute_shear_ratios_kk(self, kk_vector, kk_labels):
        """
        Compute weak lensing shear ratios R_ij / R_ik = Cl^{κ_i κ_j} / Cl^{κ_i κ_k}
        for fixed i and j > i, k > j.

        Parameters
        ----------
        kk_vector : np.ndarray
            Flattened Cl^{κ_i κ_j} vector from extract_kk_vector().
        kk_labels : list of str
            Labels like 'kk_0_1', 'kk_0_2', etc. with i <= j.

        Returns
        -------
        R : np.ndarray
            Shear ratio vector.
        ratio_labels : list of tuple
            Triplets (i, j, k) corresponding to R_{ij}/R_{ik}.
        """
        import re
        from itertools import combinations

        # Parse (i, j) pairs
        pairs = []
        for label in kk_labels:
            i, j = map(int, re.findall(r'\d+', label))
            pairs.append((i, j))  # zero-based

        # Map from (i, j) to index in kk_vector
        pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}

        R = []
        ratio_labels = []

        for i in range(len(self.nz_src)):
            # All j > i with (i, j) in pairs
            valid_js = [j for (ii, j) in pairs if ii == i and j > i]
            for j, k in combinations(valid_js, 2):
                num = kk_vector[pair_to_index[(i, j)]]
                den = kk_vector[pair_to_index[(i, k)]]
                R.append(num / den)
                ratio_labels.append((i + 1, j + 1, k + 1))  # 1-based for readability

        R = np.array(R)
        return R, ratio_labels

    def sample_kk_vector(self, kk_vector, kk_cov, nsamples=20000, seed=None):
        """
        Draw Monte Carlo realizations of the kk_vector from its covariance.

        Parameters
        ----------
        kk_vector : np.ndarray
            Mean vector of C_l^{κ_i κ_j} values (shape: [n, n_ell]).
        kk_cov : np.ndarray
            Covariance matrix of the kk_vector (shape: [n, n]).
        nsamples : int
            Number of Monte Carlo samples to generate.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        samples : np.ndarray
            Noisy samples of kk_vector (shape: [nsamples, n, n_ell]).
        """
        rng = np.random.default_rng(seed)
        n, n_ell = kk_vector.shape
        mean = kk_vector[:, 0]  # assume single ℓ bin
        samples = rng.multivariate_normal(mean, kk_cov, size=nsamples)
        samples = samples[..., np.newaxis] * np.ones((1, 1, n_ell))
        return samples

    def monte_carlo_shear_ratio_cov(self, kk_vector, kk_cov, kk_labels, nsamples=20000, seed=None):
        """
        Compute the covariance of the shear ratio data vector using MC sampling
        for κκ angular power spectra.

        Parameters
        ----------
        kk_vector : np.ndarray
            Mean kk vector, shape (n_valid, n_ell).
        kk_cov : np.ndarray
            Covariance matrix of kk vector, shape (n_valid, n_valid).
        kk_labels : list of str
            Labels for the kk vector elements.
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
        samples = self.sample_kk_vector(kk_vector, kk_cov, nsamples=nsamples, seed=seed)
        n_ell = kk_vector.shape[1]

        R_samples = []
        for s in range(nsamples):
            R, labels = self.compute_shear_ratios_kk(samples[s], kk_labels)
            R_samples.append(R[:, 0])  # assuming single ℓ bin

        R_samples = np.array(R_samples)  # shape (nsamples, n_ratios)
        R_mean = np.mean(R_samples, axis=0)
        R_cov = np.cov(R_samples, rowvar=False)

        return R_cov, R_mean, labels



    def compute_shear_ratio_derivatives(self, cl_dict, kk_labels, param_dict, step_frac=0.05):
        """
        Compute numerical derivatives of the shear-ratio data vector R = Cl^{kk}_ij / Cl^{kk}_ik
        with respect to cosmological and nuisance parameters, for κκ spectra.

        Parameters
        ----------
        cl_dict : dict
            Dictionary of fiducial Cls.
        kk_labels : list of str
            Labels for the Cl^{kk} vector.
        param_dict : dict
            Dictionary of parameter names and fiducial values.
        step_frac : float
            Fractional step size for finite difference derivatives.

        Returns
        -------
        derivs : dict
            Dictionary of derivatives dR/dp.
        R_fid : np.ndarray
            Fiducial shear ratio vector.
        ratio_labels : list of tuple
            List of (i, j, k) for each shear ratio.
        """
        derivs = {}
        kk_vector_fid, _ = self.extract_kk_vector(cl_dict)
        R_fid, ratio_labels = self.compute_shear_ratios_kk(kk_vector_fid, kk_labels)

        fid_copy = self.fid_cosmo.copy()
        nz_src_copy = [(z.copy(), nz.copy()) for (z, nz) in self.nz_src]

        # --- cosmological parameters ---
        for p, val in param_dict.items():
            dval = val * step_frac

            self.fid_cosmo = fid_copy.copy()
            self.fid_cosmo[p] = val + dval
            self._init_cosmology()
            cl_plus = self.compute_cls()
            kk_plus, _ = self.extract_kk_vector(cl_plus)
            R_plus, _ = self.compute_shear_ratios_kk(kk_plus, kk_labels)

            self.fid_cosmo = fid_copy.copy()
            self.fid_cosmo[p] = val - dval
            self._init_cosmology()
            cl_minus = self.compute_cls()
            kk_minus, _ = self.extract_kk_vector(cl_minus)
            R_minus, _ = self.compute_shear_ratios_kk(kk_minus, kk_labels)

            derivs[p] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dval)

        # --- delta_z_src nuisance parameters ---
        dz = 0.002
        if self.marginalize_delta_z_src:
            for i in range(len(self.nz_src)):
                param_name = f"delta_z_src_{i}"

                self.nz_src = [(z + (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_src_copy)]
                self._init_cosmology()
                cl_plus = self.compute_cls()
                kk_plus, _ = self.extract_kk_vector(cl_plus)
                R_plus, _ = self.compute_shear_ratios_kk(kk_plus, kk_labels)

                self.nz_src = [(z - (dz if idx == i else 0), nz.copy()) for idx, (z, nz) in enumerate(nz_src_copy)]
                self._init_cosmology()
                cl_minus = self.compute_cls()
                kk_minus, _ = self.extract_kk_vector(cl_minus)
                R_minus, _ = self.compute_shear_ratios_kk(kk_minus, kk_labels)

                derivs[param_name] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dz)

        # --- multiplicative bias nuisance parameters ---
        dm = 0.005
        if self.marginalize_multiplicative_bias:
            for i in range(len(self.nz_src)):
                param_name = f"m_src_{i}"

                kk_vector_perturb = kk_vector_fid.copy()
                for j, label in enumerate(kk_labels):
                    src_i, src_j = map(int, label.split('_')[1:])
                    if src_i == i or src_j == i:
                        kk_vector_perturb[j] = kk_vector_fid[j] * (1 + dm)
                R_plus, _ = self.compute_shear_ratios_kk(kk_vector_perturb, kk_labels)

                kk_vector_perturb = kk_vector_fid.copy()
                for j, label in enumerate(kk_labels):
                    src_i, src_j = map(int, label.split('_')[1:])
                    if src_i == i or src_j == i:
                        kk_vector_perturb[j] = kk_vector_fid[j] * (1 - dm)
                R_minus, _ = self.compute_shear_ratios_kk(kk_vector_perturb, kk_labels)

                derivs[param_name] = (R_plus[:, 0] - R_minus[:, 0]) / (2 * dm)

        # --- restore fiducial state ---
        self.fid_cosmo = fid_copy.copy()
        self.nz_src = nz_src_copy
        self._init_cosmology()

        return derivs, R_fid[:, 0], ratio_labels

    def compute_shear_ratio_fisher(self, R_cov, derivs):
        """
        Compute Fisher matrix from κκ shear ratio derivatives and covariance.
        Adds Gaussian priors for delta_z_src and multiplicative bias if marginalized.

        Parameters
        ----------
        R_cov : np.ndarray
            Covariance matrix of the κκ shear ratio data vector.
        derivs : dict
            Dictionary of derivatives of the shear ratio vector w.r.t. each parameter.

        Returns
        -------
        fisher : np.ndarray
            Fisher matrix for the parameters.
        param_list : list of str
            Names of the parameters corresponding to the Fisher matrix.
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

        # Add Gaussian priors for delta_z_src
        if self.marginalize_delta_z_src:
            for i in range(len(self.nz_src)):
                param = f"delta_z_src_{i}"
                if param in param_list:
                    prior_std = self.delta_z_src_prior_std
                    idx = param_list.index(param)
                    fisher[idx, idx] += 1.0 / prior_std**2

        # Add Gaussian priors for multiplicative bias
        if self.marginalize_multiplicative_bias:
            for i in range(len(self.nz_src)):
                param = f"m_src_{i}"
                if param in param_list:
                    prior_std = self.multiplicative_bias_prior_std
                    idx = param_list.index(param)
                    fisher[idx, idx] += 1.0 / prior_std**2

        return fisher, param_list

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



