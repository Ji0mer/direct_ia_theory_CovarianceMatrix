import os
import time
import numpy as np
from scipy.integrate import simpson
from scipy.special import jn
from concurrent.futures import ProcessPoolExecutor, as_completed

class Compute_covmat():
    def __init__(self,nzs,nzs_bin,nzd,nzd_bin,sigma_e,rbins,rres,kuse,zuse,pgi,pii,pgg,nv=[0,2,[0,4]],logspace=True,avg_jn=True,load_data=False,path=None):
        
        self.rbins = rbins
        self.res = rres
        self.ktemp = kuse
        self.k = kuse
        self.z = zuse
        self.nv = nv
        self.rp = {}
        self.j = {}
        self.avg_jn = avg_jn
        
        if load_data == False:
            self.set_jn_data()
        elif load_data == True:
            if path == None:
                self.load_jn_data()
            else:
                self.load_data(path)
        
        self.pgi = pgi
        self.pii = pii
        self.pgg = pgg
        
        self.nzs = nzs
        self.nzs_bin = nzs_bin
        self.nzd = nzd
        self.nzd_bin = nzd_bin
        self.sigma_e = sigma_e
        
        # Pre-calculate 2D noise terms
        # Shape: (1, nz) -> Broadcastable to (nk, nz) if needed, or just (nz,)
        # 原代码中 pgi 是 (nz, nk)，所以这里的 noise 需要匹配维度进行加法
        self.nzs_2d = np.nan_to_num(self.sigma_e**2/self.nzs, nan=0.0, posinf=0.0, neginf=0.0)[:, np.newaxis]
        self.nzs_bin_2d = np.nan_to_num(self.sigma_e**2/self.nzs_bin, nan=0.0, posinf=0.0, neginf=0.0)[:, np.newaxis]
        
        self.nzd_2d = np.nan_to_num(1/self.nzd, nan=0.0, posinf=0.0, neginf=0.0)[:, np.newaxis]
        self.nzd_bin_2d = np.nan_to_num(1/self.nzd_bin, nan=0.0, posinf=0.0, neginf=0.0)[:, np.newaxis]
        
    # ... [set_jn_data, save_jn_data, load_jn_data, compute_jn 等函数保持不变，此处省略以节省空间] ...
    # 请保留原有的贝塞尔函数计算和加载代码
    
    def set_jn_data(self):
        # (保留原代码)
        print("Compute Bessel function parallel...")
        if self.avg_jn == True:
            tasks = []
            keys = []
            for i in self.nv:
                if isinstance(i, list):
                    tasks.append(('avg_jns', i))
                    keys.append(str(i))
                elif isinstance(i, int):
                    tasks.append(('avg_jn', i))
                    keys.append(i)
                else:
                    self.rp[i], self.j[i] = 0, 0
            if tasks:
                max_workers = min(len(tasks), os.cpu_count())
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_key = {}
                    for task, key in zip(tasks, keys):
                        if task[0] == 'avg_jns':
                            future = executor.submit(self.compute_avg_jns, task[1])
                        else:
                            future = executor.submit(self.compute_avg_jn, task[1])
                        future_to_key[future] = key
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            rp_result, j_result = future.result()
                            self.rp[key] = rp_result
                            self.j[key] = j_result
                        except Exception as exc:
                            print(f'Task {key} generated an exception: {exc}')
                            self.rp[key], self.j[key] = 0, 0
        else:
            for i in self.nv:
                if isinstance(i, int):
                    self.rp[i], self.j[i] = self.compute_jn(i)
                elif isinstance(i, list):
                    self.rp[str(i)], self.j[i] = self.compute_jns(i)
                else:
                    self.rp[i], self.j[i] = 0, 0
        return 0

    def load_jn_data(self, file_path="/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_theory/output/avg_jn_nz/",numbins=20):
        print("Only using saved k, rp, averaged jn....")
        self.k = np.load(file_path+"k.npy")
        self.rp[0] = np.load(file_path+"rp_nv0.npy")
        self.rp[2] = np.load(file_path+"rp_nv2.npy")
        self.rp["[0, 4]"] = np.load(file_path+"rp_nv04.npy")
        j0 = {}
        j2 = {}
        j04 = {}
        for i in range(numbins):
            j0[i] = np.load(file_path+"j0_"+str(i)+".npy")
            j2[i] = np.load(file_path+"j2_"+str(i)+".npy")
            j04[i] = np.load(file_path+"j04_"+str(i)+".npy")
        self.j[0] = j0
        self.j[2] = j2
        self.j["[0, 4]"] = j04
        return True

    def compute_jn(self,nvi):
        j = {}
        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        for ind1 in range( len(rnew) ):
            kr = self.k*rnew[ind1]
            j[ind1] = jn(nvi,kr)
        return rnew,j
    
    def compute_jns(self,nvi):
        j = {}
        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        for ind1 in range( len(rnew) ):
            kr = self.k*rnew[ind1]
            sum_jns = np.zeros_like(kr)
            for ind2 in nvi:
                sum_jns += jn(ind2,kr)
            j[ind1] = sum_jns
        return rnew,j
    
    def compute_avg_jn(self,nvi):
        avg_j = {}
        for i in range( len(self.rbins)-1 ):
            if self.rbins[i+1] < 1:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res/5 )
            else:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res )
            kr = np.outer( self.k,ruse )
            avg_jn = simpson( 2*np.pi*ruse*jn(nvi,kr),x=ruse )
            avg_jn /= np.pi*(np.max(ruse)**2 - np.min(ruse)**2)
            avg_j[i] = avg_jn
        rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
        return rnew,avg_j
    
    def compute_avg_jns(self,nvi):
        avg_j = {}
        for i in range( len(self.rbins)-1 ):
            if self.rbins[i+1] < 1:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res/5 )
            else:
                ruse = np.arange( self.rbins[i],self.rbins[i+1],self.res )
            kr = np.outer( self.k,ruse )
            sum_jn = np.zeros_like( kr )
            for j in nvi:
                sum_jn += jn(j,kr)
            avg_jn = simpson( 2*np.pi*ruse*sum_jn,x=ruse )
            avg_jn /= simpson( 2*np.pi*ruse,x=ruse )
            avg_j[i] = avg_jn
        rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
        return rnew,avg_j

    # =========================================================
    # Core Vectorized Computation Logic
    # =========================================================

    def _compute_vectorized(self, zuse, W1, W2, W1_bin, W2_bin, 
                            P_signal_z, P_noise_z, 
                            jn_idx_1, jn_idx_2, nbins):
        """
        Internal helper to vectorize the double loop integration.
        P_signal_z: (nz, nk) array for off-diagonal terms
        P_noise_z:  (nz, nk) array for diagonal terms (usually full power including noise)
        """
        # 1. Calculate Normalization Factors
        # W1, W2 shape: (nz,)
        W1_int = np.trapz(W1, zuse)
        W2_int = np.trapz(W2, zuse)
        norm = W1_int * W2_int

        # 2. Perform Z-Integration ONCE (Vectorized over k)
        # W1_bin, W2_bin shape: (nz,) -> reshape to (nz, 1) for broadcasting
        W_kernel = (W1_bin * W2_bin)[:, np.newaxis] # (nz, 1)
        
        # Integrate P(k, z) over z
        # Result shape: (nk,)
        Pk_integrated_signal = np.trapz(W_kernel * P_signal_z, zuse, axis=0) / norm
        Pk_integrated_noise  = np.trapz(W_kernel * P_noise_z,  zuse, axis=0) / norm

        # 3. Prepare K-Integration
        # Integrand = k/(2pi) * P(k)_integrated
        integrand_signal = self.k / (2 * np.pi) * Pk_integrated_signal
        integrand_noise  = self.k / (2 * np.pi) * Pk_integrated_noise

        # 4. Construct Bessel Matrices
        # Shape: (nbins, nk)
        J1 = np.array([self.j[jn_idx_1][i] for i in range(nbins)])
        J2 = np.array([self.j[jn_idx_2][i] for i in range(nbins)])

        # 5. Perform K-Integration using Matrix Broadcasting (Vectorized Simpson)
        # We want Cov[i,j] = Integral( J1[i] * J2[j] * Integrand )
        
        # Broadcast shapes: (nbins, 1, nk) * (1, nbins, nk) * (1, 1, nk) -> (nbins, nbins, nk)
        term_signal = J1[:, None, :] * J2[None, :, :] * integrand_signal[None, None, :]
        
        # Integrate over last axis (k)
        cov = simpson(term_signal, x=self.k, axis=-1)

        # 6. Add Noise Correction to Diagonal
        # Diagonal needs P_noise, currently has P_signal. Add (P_noise - P_signal).
        # Only for i == j.
        diff_integrand = integrand_noise - integrand_signal
        term_diag_diff = J1 * J2 * diff_integrand[None, :] # (nbins, nk)
        diag_correction = simpson(term_diag_diff, x=self.k, axis=-1)
        
        # Add to diagonal
        rows, cols = np.diag_indices_from(cov)
        cov[rows, cols] += diag_correction

        return cov

    # =========================================================
    # Refactored Methods calling the vectorized helper
    # =========================================================

    def compute_wgpwgp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        # Off-diagonal P(k): No noise
        P_signal = self.pgg * self.nzs_bin_2d + self.pii * self.nzd_bin_2d + 2 * self.pgi**2
        # Diagonal P(k): With noise (1/n)
        P_noise  = (self.pgg + self.nzd_bin_2d) * (self.pii + self.nzs_bin_2d) + self.pgi**2
        
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_signal, P_noise, 
                                        2, 2, nbins) # j2, j2

    def compute_wgpwpp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        # P(k) is same for diag and off-diag (no auto-correlation of same tracers causing shot noise difference here typically, 
        # or noise is negligible/handled differently. Based on original code, diag uses same formula).
        P_common = 2 * self.pgi * (self.pii + self.nzs_bin_2d)
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        2, "[0, 4]", nbins)

    def compute_wgpwgg(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_common = 2 * self.pgi * (self.pgg + self.nzd_bin_2d)
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        2, 0, nbins)

    def compute_wppwgp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_common = 2 * (self.pii + self.nzs_bin_2d) * self.pgi
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        "[0, 4]", 2, nbins)

    def compute_wppwpp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_signal = 2 * (self.pii**2 + 2 * self.pii * self.nzs_bin_2d)
        P_noise  = 2 * (self.pii + self.nzs_bin_2d)**2
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_signal, P_noise, 
                                        "[0, 4]", "[0, 4]", nbins)

    def compute_wppwgg(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_common = 2 * self.pgi**2
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        "[0, 4]", 0, nbins)

    def compute_wggwgp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_common = 2 * (self.pgg + self.nzd_bin_2d) * self.pgi
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        0, 2, nbins)

    def compute_wggwpp(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_common = 2 * self.pgi**2
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_common, P_common, 
                                        0, "[0, 4]", nbins)

    def compute_wggwgg(self, zuse, W1, W2, W1_bin, W2_bin, nbins=20):
        P_signal = 2 * (self.pgg**2 + 2 * self.pgg * self.nzd_bin_2d)
        P_noise  = 2 * (self.pgg + self.nzd_bin_2d)**2
        return self._compute_vectorized(zuse, W1, W2, W1_bin, W2_bin, 
                                        P_signal, P_noise, 
                                        0, 0, nbins)