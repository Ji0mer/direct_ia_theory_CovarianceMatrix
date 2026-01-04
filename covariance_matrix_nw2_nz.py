import os
import sys
import time
import numpy as np
from dht_simpson_nw2_nz import Compute_covmat
from scipy import integrate
import scipy.integrate as sint
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from cosmosis.datablock import names, option_section
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 #use Planck15 if you can
import astropy.units as u

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new

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

def setup(options):

    sample = options.get_string(option_section,"sample",default="forecast_sample")
    zmin = options.get_double(option_section, "zmin",default=0.4)
    zmax = options.get_double(option_section, "zmax",default=0.6)
    sigma_e = options.get_double(option_section,"sigma_e",default=0.25)
    area_shape = options.get_double(option_section, "area_shape", default=600.0)
    area_dens = options.get_double(option_section, "area_dens", default=1000.0)
    rmin = options.get_double(option_section,"rmin",default=0.1)
    rmax = options.get_double(option_section,"rmax",default=350.0)
    nr = options.get_int(option_section,"nr",default=21)
    nz_factor = options.get_double(option_section,"nz_factor",default=1.0)
    
    Pi_max = options.get_double(option_section,"Pi_max",default=100.0)
    
    return sample, zmin, zmax, sigma_e, area_shape, area_dens, rmin, rmax, nr, nz_factor, Pi_max


def execute(block, config):

    sample, zmin, zmax, sigma_e, area_shape, area_dens, rmin, rmax, nr, nz_factor, Pi_max = config
    
    rbins = np.logspace( np.log10(rmin),np.log10(rmax),nr )
    
    # load linear matter power spectrum
    plin = block['matter_power_lin','p_k']
    z = block['matter_power_lin','z']
    kh = block['matter_power_lin','k_h']
    # load n(z)
    zs = block['nz_'+sample+"_shape", 'z']
    nzs = block['nz_'+sample+"_shape", 'raw_all']*nz_factor
    nzs_bin = block['nz_'+sample+"_shape", 'raw']*nz_factor
    zd = block['nz_'+sample+"_density", 'z']
    nzd = block['nz_'+sample+"_density", 'raw_all']*nz_factor
    nzd_bin = block['nz_'+sample+"_density", 'raw']*nz_factor
    
    # interpolate linear matter power spectrum and n(z) into kuse,zuse
    kuse = np.logspace( np.log10( np.min(4.999999771825967e-05) ),np.log10(np.max(344.99999999999994)),9001 )
    zuse = np.linspace( np.min(z[0])+1e-6,np.max(z[-1]),401 )
    
    pml_kinterp = interp1d( kh,plin,axis=1,bounds_error=False,fill_value=0 )
    pml_temp = pml_kinterp(kuse)
    pml_zinterp = interp1d( z,pml_temp,axis=0 )
    pm_lin = pml_zinterp(zuse)
    
    #nzs
    nzs_interp = interp1d( zs,nzs,bounds_error=False,fill_value=0 )
    nzs = nzs_interp( zuse )
    nzs_bin_interp = interp1d( zs,nzs_bin,bounds_error=False,fill_value=0 )
    nzs_bin = nzs_bin_interp( zuse )
    # nzd
    nzd_interp = interp1d( zd,nzd,bounds_error=False,fill_value=0 )
    nzd = nzd_interp( zuse )
    nzd_bin_interp = interp1d( zd,nzd_bin,bounds_error=False,fill_value=0 )
    nzd_bin = nzd_bin_interp( zuse )
    
    # compute comoving volume
    h0 = block["cosmological_parameters","h0"]
    cosmo=Planck13.clone(H0=h0*100)
    # 转成球面立体角 Ω（单位 sr）
    Omega_shape = (area_shape * u.deg**2).to(u.sr)
    V_allsky_zmax = cosmo.comoving_volume(zmax)      # [Mpc^3]
    V_allsky_zmin = cosmo.comoving_volume(zmin)      # [Mpc^3]
    V_shell = (V_allsky_zmax - V_allsky_zmin) * (Omega_shape / (4*np.pi * u.sr))  # [Mpc^3]
    # convert to Mpc^3/h^3
    V_shell_shape = V_shell.value * h0**3
    
    Omega_density = (area_dens * u.deg**2).to(u.sr)
    V_allsky_zmax = cosmo.comoving_volume(zmax)      # [Mpc^3]
    V_allsky_zmin = cosmo.comoving_volume(zmin)      # [Mpc^3]
    V_shell = (V_allsky_zmax - V_allsky_zmin) * (Omega_density / (4*np.pi * u.sr))  # [Mpc^3]
    # convert to Mpc^3/h^3
    V_shell_density = V_shell.value * h0**3
    
    # compute w_dd, w_ds, w_ss
    z_chi = block["distances","z"]
    chi = block["distances","d_m"]
    chi_interp = interp1d(z_chi,chi)
    Chi = chi_interp(zuse)
    
    dchidz = dxdz = np.gradient(Chi,zuse[1]-zuse[0])
    W_ds = nzs * nzd /Chi/Chi/dchidz
    W_ds_bin = nzs_bin * nzd_bin /Chi/Chi/dchidz
    W_dd = nzd * nzd /Chi/Chi/dchidz
    W_dd_bin = nzd_bin * nzd_bin /Chi/Chi/dchidz
    W_ss = nzs * nzs /Chi/Chi/dchidz
    W_ss_bin = nzs_bin * nzs_bin /Chi/Chi/dchidz
    
    # compute Dz
    # use ind to handle mild scale-dependence in growth
    ind = np.where(kuse > 0.03)[0][0]
    Dz = np.sqrt(pm_lin[:, ind] / pm_lin[0, ind])
    Dz_interp = interp1d(zuse,Dz)
    Dzeff = Dz_interp(zuse)
    
    # compute C1
    A1 = block["intrinsic_alignment_parameters","A1"]
    b1 = block['bias_%s_density'%sample,"b1E_bin1"]
    C1 = compute_c1(A1,Dzeff,0.5)
    
    # compute Pg+,P++,Pgg
    pgi = np.zeros_like(pm_lin)
    pii = np.zeros_like(pm_lin)
    for i in range( len(zuse) ):
        pgi[i] = b1*C1[i]*pm_lin[i]*0
        pii[i] = C1[i]**2*pm_lin[i]*0
    pgg = b1**2*pm_lin*0
    
    cc = cc = Compute_covmat(nzs,nzs_bin,nzd,nzd_bin,sigma_e,rbins,1e-3,kuse,zuse,pgi,pii,pgg,nv=[0,2,[0,4]],load_data = True)
    
    
    cov_gpgp = cc.compute_wgpwgp(zuse,W_ds,W_ds,W_ds_bin,W_ds_bin)
    cov_gppp = cc.compute_wgpwpp(zuse,W_ds,W_ss,W_ds_bin,W_ss_bin)
    cov_gpgg = cc.compute_wgpwgg(zuse,W_ds,W_dd,W_ds_bin,W_dd_bin)
    
    cov_ppgp = cc.compute_wppwgp(zuse,W_ss,W_ds,W_ss_bin,W_ds_bin)
    cov_pppp = cc.compute_wppwpp(zuse,W_ss,W_ss,W_ss_bin,W_ss_bin)
    cov_ppgg = cc.compute_wppwgg(zuse,W_ss,W_dd,W_ss_bin,W_dd_bin)
    
    cov_gggp = cc.compute_wggwgp(zuse,W_dd,W_ds,W_dd_bin,W_ds_bin)
    cov_ggpp = cc.compute_wggwpp(zuse,W_dd,W_ss,W_dd_bin,W_ss_bin)
    cov_gggg = cc.compute_wggwgg(zuse,W_dd,W_dd,W_dd_bin,W_dd_bin)
    
    

    clen = len(cc.rp[0])
    Cov = np.zeros( (3*clen,3*clen) )
    
    
    Cov[0*clen:1*clen,0*clen:1*clen] += cov_gpgp*2*np.pi*Pi_max/V_shell_shape
    Cov[0*clen:1*clen,1*clen:2*clen] += cov_gppp*2*np.pi*Pi_max/V_shell_shape
    Cov[0*clen:1*clen,2*clen:3*clen] += cov_gpgg*2*np.pi*Pi_max/V_shell_shape
    
    Cov[1*clen:2*clen,0*clen:1*clen] += cov_ppgp*2*np.pi*Pi_max/V_shell_shape
    Cov[1*clen:2*clen,1*clen:2*clen] += cov_pppp*2*np.pi*Pi_max/V_shell_shape
    Cov[1*clen:2*clen,2*clen:3*clen] += cov_ppgg*2*np.pi*Pi_max/V_shell_shape
    
    Cov[2*clen:3*clen,0*clen:1*clen] += cov_gggp*2*np.pi*Pi_max/V_shell_shape
    Cov[2*clen:3*clen,1*clen:2*clen] += cov_ggpp*2*np.pi*Pi_max/V_shell_shape
    Cov[2*clen:3*clen,2*clen:3*clen] += cov_gggg*2*np.pi*Pi_max/V_shell_density
    
    
    block["covmat","Cov"] = Cov
    block["covmat","rp0"] = cc.rp[0]
    block["covmat","rp2"] = cc.rp[2]
    block["covmat","rp04"] = cc.rp["[0, 4]"]
    print(cc.rp[0])

    return 0













































