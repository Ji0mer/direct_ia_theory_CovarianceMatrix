import os
import time
import itertools
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import simpson,quad
from scipy.interpolate import interp1d
from scipy.special import jn, jn_zeros,jv
#from scipy.integrate import trapezoid as trapz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement

#############################################################################################################################################
#############################################################################################################################################

def ht(nv,k,fk,rout,kres = 5e-4):
    interp_fk = interp1d(k,fk)
    knew = np.arange( k[0],k[-1],kres )
    fknew = interp_fk( knew )
    kr = np.outer( rout,knew )
    j = jn(nv,kr)
    Fr = simpson(j*fknew*knew/(2*np.pi),x=knew)
    return Fr

def iht(nv,r,Fr,kout,rres=5e-4):
    interp_Fr = interp1d(r,Fr)
    rnew = np.arange( r[0],r[-1],rres )
    Frnew = interp_Fr( rnew )
    kr = np.outer( kout,rnew )
    j = jn(nv,kr)
    fk = simpson(j*Frnew*rnew*2*np.pi,x=rnew)
    return fk

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new

# Copied from CosmoSIS.
def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  #  h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI ** 2 / (8 * np.pi * G)  #  h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f

def compute_c1(A1,Dz,z_out,z_piv=0,alpha1=0,Omega_m=0.3):
    C1_RHOCRIT = compute_c1_baseline()
    return -1.0*A1*C1_RHOCRIT*Omega_m/Dz*( (1.0+z_out)/(1.0+z_piv) )**alpha1

#############################################################################################################################################
#############################################################################################################################################


#############################################################################################################################################
#############################################################################################################################################

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
        else:
            pass
        self.pgi = pgi
        self.pii = pii
        self.pgg = pgg
        
        self.nzs = nzs
        self.nzs_bin = nzs_bin
        self.nzd = nzd
        self.nzd_bin = nzd_bin
        self.sigma_e = sigma_e
        
        self.nzs_2d = np.nan_to_num(self.sigma_e**2/self.nzs, nan=0.0, posinf=0.0, neginf=0.0)
        nzs_new_axis = self.nzs_2d[:,np.newaxis]
        self.nzs_bin_2d = np.nan_to_num(self.sigma_e**2/self.nzs_bin, nan=0.0, posinf=0.0, neginf=0.0)
        nzs_bin_new_axis = self.nzs_bin_2d[:,np.newaxis]
        
        self.zeros_row = np.zeros_like(self.k)
        self.nzs_2d = nzs_new_axis + self.zeros_row
        self.nzs_bin_2d = nzs_bin_new_axis + self.zeros_row
        
        self.nzd_2d = np.nan_to_num(1/self.nzd, nan=0.0, posinf=0.0, neginf=0.0)
        nzd_new_axis = self.nzd_2d[:,np.newaxis]
        self.nzd_bin_2d = np.nan_to_num(1/self.nzd_bin, nan=0.0, posinf=0.0, neginf=0.0)
        nzd_bin_new_axis = self.nzd_bin_2d[:,np.newaxis]
        
        self.nzd_2d = nzd_new_axis + self.zeros_row
        self.nzd_bin_2d = nzd_bin_new_axis + self.zeros_row
        
    def save_jn_data(self,file_path="/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_theory/output/avg_jn_nz/"):
        os.makedirs(file_path, exist_ok=True)
        np.save(file_path+"k.npy",self.k)
        np.save(file_path+"rbins.npy",self.rbins)

        np.save(file_path+"rp_nv0.npy",self.rp[0])
        np.save(file_path+"rp_nv2.npy",self.rp[2])
        np.save(file_path+"rp_nv04.npy",self.rp["[0, 4]"])


        for i in range( len(self.j[0]) ):
            file_name = file_path+"j0_"+str(i)+".npy"
            np.save(file_name,self.j[0][i])

        for i in range( len(self.j[2]) ):
            file_name = file_path+"j2_"+str(i)+".npy"
            np.save(file_name,self.j[2][i])

        for i in range( len(self.j["[0, 4]"]) ):
            file_name = file_path+"j04_"+str(i)+".npy"
            np.save(file_name,self.j["[0, 4]"][i])
    
    def set_jn_data(self):
        
        print("Compute Bessel function parallel...")
    
        if self.avg_jn == True:
            # 准备并行计算的任务
            tasks = []
            keys = []
            
            for i in self.nv:
                if isinstance(i, list):
                    print(i)
                    tasks.append(('avg_jns', i))
                    keys.append(str(i))
                elif isinstance(i, int):
                    print(i)
                    tasks.append(('avg_jn', i))
                    keys.append(i)
                else:
                    print(i)
                    # 直接设置为0，不需要计算
                    self.rp[i], self.j[i] = 0, 0
            
            # 并行计算
            if tasks:
                # 使用ProcessPoolExecutor (推荐)
                max_workers = min(len(tasks), os.cpu_count())
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 提交任务
                    future_to_key = {}
                    for task, key in zip(tasks, keys):
                        if task[0] == 'avg_jns':
                            future = executor.submit(self.compute_avg_jns, task[1])
                        else:  # avg_jn
                            future = executor.submit(self.compute_avg_jn, task[1])
                        future_to_key[future] = key
                    
                    # 收集结果
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
            # 非平均贝塞尔函数的计算保持原样（通常较快）
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
        #self.k *= ( 1-np.heaviside(self.k-20,0) )
        
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
        #rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
        rnew = ( self.rbins[:-1]+self.rbins[1:] )/2
        for ind1 in range( len(rnew) ):
            kr = self.k*rnew[ind1]
            j[ind1] = jn(nvi,kr)
        
        return rnew,j
    
    def compute_jns(self,nvi):
        j = {}
        #rnew = np.sqrt( self.rbins[:-1]*self.rbins[1:] )
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
    
    # g+
    def compute_wgpwgp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                if i == j:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( (self.pgg+self.nzd_bin_2d)*(self.pii+self.nzs_bin_2d)+self.pgi**2 ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*power_componments,x=self.k )
                else:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( self.pgg*self.nzs_bin_2d+self.pii*self.nzd_bin_2d+2*self.pgi**2 ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[2][j]*power_componments,x=self.k )
        return cov

    def compute_wgpwpp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*self.pgi*(self.pii+self.nzs_bin_2d) ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j["[0, 4]"][j]*power_componments,x=self.k )
        return cov
    
    def compute_wgpwgg(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*self.pgi*(self.pgg+self.nzd_bin_2d) ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[2][i]*self.j[0][j]*power_componments,x=self.k )
        return cov
    
    # ++
    def compute_wppwgp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pii+self.nzs_bin_2d)*self.pgi ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j[2][j]*power_componments,x=self.k )
        return cov

    def compute_wppwpp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                if i == j:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pii+self.nzs_bin_2d)*(self.pii+self.nzs_bin_2d) ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j["[0, 4]"][j]*power_componments,x=self.k )
                else:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pii**2+2*self.pii*self.nzs_bin_2d) ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j["[0, 4]"][j]*power_componments,x=self.k )
        return cov
    
    def compute_wppwgg(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*self.pgi**2 ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j["[0, 4]"][i]*self.j[0][j]*power_componments,x=self.k )
        return cov
    
    # gg
    def compute_wggwgp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pgg+self.nzd_bin_2d)*self.pgi ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[2][j]*power_componments,x=self.k )
        return cov

    def compute_wggwpp(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*self.pgi**2 ),zuse,axis=0 )/( W12_int )
                cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j["[0, 4]"][j]*power_componments,x=self.k )
        return cov
    
    def compute_wggwgg(self,zuse,W1,W2,W1_bin,W2_bin,nbins=20):
        
        cov = np.zeros( (nbins,nbins) )
        
        W1_new_axis = W1[:,np.newaxis]
        W1_2d = W1_new_axis + self.zeros_row
        W1_int = np.trapz( W1_2d,zuse,axis=0 )
    
        W2_new_axis = W2[:,np.newaxis]
        W2_2d = W2_new_axis + self.zeros_row
        W2_int = np.trapz( W2_2d,zuse,axis=0 )
        
        W12_int = np.trapz( W1_2d*W2_2d,zuse,axis=0 )
        
        W1_bin_new_axis = W1_bin[:,np.newaxis]
        W1_bin_2d = W1_bin_new_axis + self.zeros_row
        W1_bin_int = np.trapz( W1_bin_2d,zuse,axis=0 )
    
        W2_bin_new_axis = W2_bin[:,np.newaxis]
        W2_bin_2d = W2_bin_new_axis + self.zeros_row
        W2_bin_int = np.trapz( W2_bin_2d,zuse,axis=0 )
    
        for i in range( nbins ):
            for j in range( nbins ):
                if i == j:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pgg+self.nzd_bin_2d)*(self.pgg+self.nzd_bin_2d) ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[0][j]*power_componments,x=self.k )
                else:
                    power_componments = np.trapz( W1_bin_2d*W2_bin_2d*( 2*(self.pgg**2+2*self.pgg*self.nzd_bin_2d) ),zuse,axis=0 )/( W12_int )
                    cov[i,j] = simpson( self.k/(2*np.pi)*self.j[0][i]*self.j[0][j]*power_componments,x=self.k )
        return cov