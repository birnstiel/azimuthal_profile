#!/usr/bin/env python
# coding: utf-8

"""Azimuthal Dust Profiles Widget"""

import pkg_resources
import os

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

from scipy.constants import golden as gr
from scipy.interpolate import interp2d

import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

import azimuthal_profile as az
import dsharp_opac as op

matplotlib.rcParams.update({'font.size': 8})

au = c.au.cgs.value
c_light = c.c.cgs.value
jy_as = (1. * u.Jy / u.arcsec).cgs.value


class Widget():

    # Set up model parameters of a radial gas profile

    na = 100
    nr = 40
    ny = 400
    rho_s = 1.67  # dust material density [g/cm^3]
    A_gas = 1.2     # gas density contrast (peak-to-valley)
    d2g = 0.01
    alpha = 1e-3
    v_frag = 1000
    sig0 = 100.
    M_star = c.M_sun.cgs.value
    sigma_y_deg = 10.     # azimuthal bump extent in degree
    sigma_y = sigma_y_deg / 180 * np.pi
    r = np.logspace(-1, 2, nr) * au  # radial grid
    y = np.linspace(- np.pi, np.pi, ny)
    Y = r[:, None] * y  # azimuth grid
    sig_g = sig0 / (r / r[0])                     # azimuthally avg. gas surface density
    sig_d = d2g * sig_g

    # maximum particle size, which can be
    # float: same maximum particle size everywhere
    # array: different particle size at every radius
    # None: will construct fragmentation/coagulation size distribution

    a0 = 0.1
    pa = 0
    a_max = a0 * (r / r[0])**pa

    # Nienke: temperature has minor influence, so changes due to changes
    # in R is primarily due to sigma_gas, which is scaled with R => change sig_c
    T = 50 * np.ones(nr)

    # initial slider position

    ir = nr // 2

    # which particle sizes to show

    sizes = [1e-4, 1e-2, 1e-1, 1e0]
    i_sizes = []

    def __init__(self, fwhm=False, radial_bump=False, lam_obs=np.array([0.13, 0.9])):
        self.fwhm = fwhm
        self.radial_bump = radial_bump
        self.A_r = 1.2           # radial gas density contrast
        self.r_bump = 20 * au    # position of the bump
        self.sigma_r = 5. * au  # radial bump extent in cm

        # figure setup

        self.width = 10
        self.height = self.width / gr * 2 / 3 * 1.5
        self.fig = plt.figure(figsize=(self.width, self.height))
        gs = gridspec.GridSpec(2, 3)

        self.ax0 = self.fig.add_subplot(gs[0, 0])
        self.ax1 = self.fig.add_subplot(gs[0, 1])
        self.ax2 = self.fig.add_subplot(gs[0, 2])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax5 = self.fig.add_subplot(gs[1, 2])

        self.fig.subplots_adjust(left=0.08, wspace=0.4, hspace=0.4, top=0.95, bottom=0.25, right=0.92)

        # OPACITY: Read opacity and interpolate on our grid

        self.lam_obs = lam_obs
        self.n_lam = len(lam_obs)
        self.nu_obs = c_light / lam_obs

        with np.load(op.get_datafile('ricci_compact.npz')) as fid:
            a_opac = fid['a']
            lam_opac = fid['lam']
            k_abs = fid['k_abs']
            # k_sca    = fid['k_sca']
            # g        = fid['g']
            # rho_s    = fid['rho_s']

        self.f_kappa = interp2d(np.log10(lam_opac), np.log10(a_opac), np.log10(k_abs))

        # calculations and plot initialization

        self.calculate_distri()
        self.calculate_intensity()

        self.init_plot0()
        self.init_plot1()
        self.init_plot2()
        self.init_plot3()
        self.init_plot4()
        self.init_plot5()

        slider_x0 = self.ax0.get_position().x0
        slider_y0 = 0.05
        slider_w = self.ax0.get_position().width
        slider_h = 0.04

        # radius slider

        self._ax_radius = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_radius = Slider(self._ax_radius, "radius", 0, self.nr - 1, valinit=self.ir, valfmt='%i')
        self._ax_radius.set_title("r = {:9.3e} au".format(self.r[self.ir] / au), fontsize='small')
        self._slider_radius.on_changed(self.update_r)

        # A_g slider

        self._ax_Ag = self.fig.add_axes([slider_x0, slider_y0 + slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_Ag = Slider(self._ax_Ag, "$A_g$", 1, 3, valinit=self.A_gas, valfmt='%.2f')
        self._slider_Ag.on_changed(self.update_all)

        # alpha slider

        self._ax_alpha = self.fig.add_axes([slider_x0, slider_y0 + 2 * slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_alpha = Slider(self._ax_alpha, "$\\alpha$", -4, -1, valinit=np.log10(self.alpha), valfmt='$10^{%.1f}$')
        self._slider_alpha.on_changed(self.update_all)

        # T slider

        slider_x0 = self.ax1.get_position().x0

        self._ax_T = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_T = Slider(self._ax_T, "$T$", 10, 200, valinit=self.T[self.ir], valfmt='%.0f K')
        self._slider_T.on_changed(self.update_all)

        # a0 slider

        self._ax_a0 = self.fig.add_axes([slider_x0, slider_y0 + slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_a0 = Slider(self._ax_a0, "$a_0$", -4, 1, valinit=self.a0, valfmt='$10^{%.1f} cm$')
        self._slider_a0.on_changed(self.update_all)

        # pa slider

        self._ax_pa = self.fig.add_axes([slider_x0, slider_y0 + 2 * slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_pa = Slider(self._ax_pa, "$p_a$", -4, 1, valinit=self.pa, valfmt='%.1f')
        self._slider_pa.on_changed(self.update_all)

        # sigma slider

        slider_x0 = self.ax2.get_position().x0

        self._ax_sigma = self.fig.add_axes([slider_x0, slider_y0, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_sigma = Slider(self._ax_sigma, r"$\Sigma_0$", -1, 4, valinit=np.log10(self.sig0), valfmt='$10^{%.1f}$')
        self._slider_sigma.on_changed(self.update_all)

        # d2g slider

        self._ax_d2g = self.fig.add_axes([slider_x0, slider_y0 + slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_d2g = Slider(self._ax_d2g, "d2g", -4, 0, valinit=np.log10(self.d2g), valfmt='$10^{%.1f}$')
        self._slider_d2g.on_changed(self.update_all)

        # vortex width slider

        self._ax_sigydeg = self.fig.add_axes([slider_x0, slider_y0 + 2 * slider_h, slider_w, slider_h], facecolor="lightgoldenrodyellow")
        self._slider_sigydeg = Slider(self._ax_sigydeg, "$\\Delta \\phi$", 1, 180, valinit=self.sigma_y_deg, valfmt='%.1f')
        self._slider_sigydeg.on_changed(self.update_all)

    def update_r(self, val):
        ir = int(np.floor(self._slider_radius.val))

        if ir != self.ir:
            self.ir = ir
            self._ax_radius.set_title(f'r = {self.r[ir] / au:9.3e} au')

            self.update_plot0()
            self.update_plot1()
            self.update_plot2(distri_update=False)
            self.update_plot3()
            self.update_plot4()
            self.update_plot5(distri_update=False)

    def update_all(self, val):

        self.A_gas = self._slider_Ag.val

        self.alpha = 10.**(self._slider_alpha.val)

        self.T = self._slider_T.val * np.ones(self.nr)

        self.a0 = 10.**self._slider_a0.val
        self.pa = self._slider_pa.val

        if np.isclose(self.pa, self._slider_pa.valmin):
            self.a_max = None
        else:
            self.a_max = self.a0 * (self.r / self.r[0])**self.pa

        self.sig0 = 10.**(self._slider_sigma.val)
        self.sig_g = self.sig0 / (self.r / self.r[0])

        if self.radial_bump:
            self.sig_g = self.sig_g * (1 + self.A_gas * np.exp(-(self.r - self.r_bump)**2 / (2 * self.sigma_r**2)))

        self.d2g = 10.**(self._slider_d2g.val)
        self.sig_d = self.sig_g * self.d2g

        self.sigma_y_deg = self._slider_sigydeg.val
        self.sigma_y = self.sigma_y_deg / 180 * np.pi

        self.calculate_distri()
        self.calculate_intensity()

        self.update_plot0()
        self.update_plot1()
        self.update_plot2(distri_update=True)
        self.update_plot3()
        self.update_plot4()
        self.update_plot5(distri_update=True)
        self.update_limits()

    def update_limits(self):

        self.ax0.set_ylim(self.sig_d_2D[:, :, 0].min(), self.sig_d_2D.max())
        self.ax1.set_ylim(self.I_nu[0, :, :].max() / jy_as * np.array([1e-5, 2]))
        self.ax4.set_ylim(np.array([1e-5, 2]) * self.distri.max())

        self.distri.max() * 1e-5, 2 * self.distri.max()

    def calculate_distri(self):
        """
        Calculate the 2D gas and dust distributions and the dust size distribution.
        It updates the objects A, sig_g_2D, sig_d_2D
        """

        # #### SIZE GRID
        # at least to a_max or 10 if a_max is none

        if self.a_max is None:
            A = np.logspace(-5, 1, self.na)
        else:
            A = np.logspace(-5, np.log10(np.max(self.a_max)), self.na)

        # #### GAS AZIMUTHAL PROFILE

        self.sig_g_2D = 1 + (self.A_gas - 1) * np.exp(- self.Y**2 / (2 * (self.r * self.sigma_y)**2)[:, None])
        if self.radial_bump:
            self.sig_g_2D = 1 + (
                self.A_gas *
                (np.exp(-(self.r - self.r_bump)**2 / (2 * self.sigma_r**2)))[:, None] *
                np.exp(- self.Y**2 / (2 * (self.r * self.sigma_y)**2)[:, None])
            )
        self.sig_g_2D *= (self.sig_g / self.sig_g_2D.mean(-1))[:, None]

        # #### DUST SIZE DISTRIBUTION

        if self.a_max is not None:
            print('a_max given -> making MRN size distribution')

            # make an MRN distribution up to a_max

            # distri = A[None, :]**0.5 * np.ones([self.nr, 1])
            # mask = A[None, :] > (self.a_max * np.ones_like(self.r))[:, None]
            # distri[mask] = 1e-100
            # distri *= (self.sig_d / distri.sum(-1))[:, None]
            # distri = distri.T

            A, _, distri = az.get_powerlaw_dust_distribution(self.sig_d, self.a_max, q=3.5, na=self.na, a0=1e-5, a1=1000)
            distri = distri.T

        else:
            print('no a_max given -> using equilibrium size distribution')
            distri = self.sig_d.copy()

        # #### DUST AZIMUTHAL PROFILES
        self.distri = distri
        self.A = A
        self.sig_d_2D = az.make_azim_distri(
            self.r,
            self.Y,
            self.A,
            self.T,
            self.sig_g_2D,
            self.distri,
            self.alpha,
            self.v_frag,
            self.rho_s,
            self.M_star,
        )

        if self.a_max is None:
            self.distri = self.sig_d_2D.mean(1).T

    def init_plot0(self):
        "Plot the azimuthal density distribution of some grain sizes"
        ir = self.ir

        self.i_sizes = self.A.searchsorted(np.minimum(self.sizes, self.A[-1]))

        self.lines_0 = []
        self.lines_0 += self.ax0.semilogy(self.y / np.pi * 180, self.sig_g_2D[ir, :] / 100, 'k-', label='gas / 100')
        for ia in self.i_sizes:
            self.lines_0 += self.ax0.semilogy(self.y / np.pi * 180, self.sig_d_2D[ir, :, ia], '-', label=f'{self.A[ia]:.1g} cm')

        self.ax0.set_xlim(-45, 45)
        self.ax0.set_ylim(self.sig_g.min() / 1e5, self.sig_g.max() * 1.5 / 100)
        self.ax0.set_xlabel('azimuth [degree]')
        self.ax0.set_ylabel(r'$\Sigma_\mathrm{d}$ [g cm$^{-2}$]')
        self.ax0.legend(fontsize='small', loc=1)

    def update_plot0(self):
        ir = self.ir
        self.i_sizes = self.A.searchsorted(np.minimum(self.sizes, self.A[-1]))

        self.lines_0[0].set_ydata(self.sig_g_2D[ir, :] / 100)
        for i_line, ia in enumerate(self.i_sizes):
            self.lines_0[i_line + 1].set_ydata(self.sig_d_2D[ir, :, ia])

    def init_plot1(self):
        "Intensity profile"
        ir = self.ir
        self.lines_1 = []

        for ilam in range(self.n_lam):
            # print(f'Intensity contrast (max/min) at {10 * self.lam_obs[ilam]:3.1f} mm = {self.I_nu[ilam, ir, :].max()/self.I_nu[ilam, ir, :].min():.3g}')
            self.lines_1 += self.ax1.semilogy(self.y * 180 / np.pi, self.I_nu[ilam, ir, :] / jy_as, label=f'$\\lambda = {10 * self.lam_obs[ilam]:3.1f}$ mm')

        self.ax1.set_xlabel('azimuth [degree]')
        self.ax1.set_ylabel('$I_\\nu$ [Jy/arsec]')
        self.ax1.legend()

    def update_plot1(self):
        ir = self.ir
        for ilam in range(self.n_lam):
            self.lines_1[ilam].set_ydata(self.I_nu[ilam, ir, :] / jy_as)

    def init_plot2(self):
        """plot the size distribution"""
        vmax = np.log10(2 * self.distri.max())
        self.cc2 = self.ax2.pcolormesh(self.r / au, self.A, np.log10(self.distri[:-1, :-1]), vmin=vmax - 9, vmax=vmax)
        self.lines_2 = self.ax2.axvline(self.r[self.ir] / au, c='k', ls='--', lw=1)
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        self.ax2.set_title('size distribution', c='k')
        self.ax2.set_xlabel('$r$ [au]')
        self.ax2.set_ylabel('paraticle size [cm]')

    def update_plot2(self, distri_update=False):
        if distri_update:
            try:
                self.cc2.remove()
            except ValueError:
                pass
            vmax = np.log10(2 * self.distri.max())
            self.cc2 = self.ax2.pcolormesh(self.r / au, self.A, np.log10(self.distri[:-1, :-1]), vmin=vmax - 9, vmax=vmax)
            if self.a_max is None:
                self.ax2.set_title('equilibrium size distribution', c='r')
            else:
                self.ax2.set_title('size distribution', c='k')
        self.lines_2.set_xdata([self.r[self.ir] / au, self.r[self.ir] / au])

    def init_plot3(self):
        "Contrast curve"
        St = self.lam_obs[0] * self.rho_s / (4. * self.sig_g)  # the "observed stokes number"

        if self.fwhm:
            self.lines_3 = self.ax3.loglog(St, self.I_nu[0, :, :].max(-1) / self.I_nu[0, :, :].min(-1))
            self.lines_3 += [self.ax3.axvline(St[self.ir], c='k', ls='--', lw=1)]

            self.ax3.set_xlabel('obs. Stokes number')
            self.ax3.set_ylabel('FWHM')
            self.ax3.set_xlim(sorted(self.ax3.get_xlim()))
            self.ax3.set_ylim(2e1, 4e3)

        else:
            stokes, err_st_low, err_st_up, A_d, err_A_d = np.loadtxt(
                pkg_resources.resource_filename(__name__, os.path.join('data', 'data.txt')),
                skiprows=1, usecols=(1, 2, 3, 4, 5)).T

            err_st = np.array((err_st_low, err_st_up))
            # self.ax3.scatter(stokes, A_d, color='r')
            self.ax3.errorbar(stokes, A_d, yerr=err_A_d, xerr=err_st, fmt='.', lw=0.5)

            self.lines_3 = self.ax3.loglog(St, self.I_nu[0, :, :].max(-1) / self.I_nu[0, :, :].min(-1))
            self.lines_3 += [self.ax3.axvline(St[self.ir], c='k', ls='--', lw=1)]

            self.ax3.set_xlabel('obs. Stokes number')
            self.ax3.set_ylabel('Intensity contrast')
            self.ax3.set_xlim(sorted(self.ax3.get_xlim()))
            self.ax3.set_ylim(1e0, 1e3)
        self.update_plot3()

    def update_plot3(self):
        ilam = 0

        St = self.lam_obs[ilam] * self.rho_s / (4. * self.sig_g)  # the "observed stokes number"
        if self.fwhm:

            fwhmprofile = []

            for ir in range(self.nr):
                I_profile = self.I_nu[ilam, ir, :]
                I_max = np.max(I_profile)
                I_min = np.min(I_profile)
                if(I_max < 2 * I_min):
                    F = 2 * np.pi
                else:

                    # we separate the left and right wing, both starting at the
                    # maximum to the left, decreasing to the right
                    i_max = I_profile.argmax()

                    I_left = np.roll(I_profile, i_max + 1)[0:self.ny // 2 + 1]
                    y_left = np.roll(self.y, i_max + 1)[0:self.ny // 2 + 1]
                    y_left = (y_left - y_left[0] + 2 * np.pi) % (2 * np.pi)

                    I_right = np.roll(I_profile, i_max)[-self.ny // 2:][::-1]
                    y_right = -np.roll(self.y, i_max)[-self.ny // 2:][::-1]
                    y_right = ((y_right - y_right[0] + 2 * np.pi) % (2 * np.pi))

                    # we interpolate to get the half-width half maximum for each side
                    # here we need to invert the arrays such that the Intensity is
                    # an increasing x-array and we can interpolate for the y.

                    F_left = np.interp(I_max / 2, I_left[::-1], y_left[::-1])
                    F_right = np.interp(I_max / 2, I_right[::-1], y_right[::-1])

                    F = abs(F_left) + abs(F_right)

                fwhmprofile += [F * 180 / np.pi]

            self.lines_3[0].set_data(St, fwhmprofile)
            self.lines_3[1].set_xdata([St[self.ir], St[self.ir]])
        else:
            self.lines_3[0].set_data(St, self.I_nu[0, :, :].max(-1) / self.I_nu[0, :, :].min(-1))
            self.lines_3[1].set_xdata([St[self.ir], St[self.ir]])

    def init_plot4(self):
        "plot the averaged particle size distribution"
        ir = self.ir

        self.lines_4 = self.ax4.loglog(self.A, self.distri[:, ir])
        self.ax4.set_xlim(1e-5, 1e1)
        self.ax4.set_ylim(self.distri.max() * 1e-5, 2 * self.distri.max())
        self.ax4.set_xlabel('particle size [cm]')
        self.ax4.set_ylabel(r'$\sigma$ [g cm$^{-2}$]')

    def update_plot4(self):
        self.lines_4[0].set_data(self.A, self.distri[:, self.ir])

    def init_plot5(self):
        "2D intensity"
        vmax = np.log10(self.I_nu[0, :, :].max())
        self.cc5 = self.ax5.pcolormesh(
            self.r / au,
            self.Y[0, :] / self.r[0] * 180 / np.pi,
            np.log10(self.I_nu[0, :-1, :-1]).T,
            vmin=vmax - 6,
            vmax=vmax)
        self.lines_5 = self.ax5.axvline(self.r[self.ir] / au, c='k', ls='--', lw=1)
        self.ax5.set_xscale('log')
        self.ax5.set_title('intensity profile')
        self.ax5.set_xlabel('$r$ [au]')
        self.ax5.set_ylabel('y [degree]')

    def update_plot5(self, distri_update=False):
        if distri_update:
            self.cc5.set_array(np.log10(self.I_nu[0, :-1, :-1]).T.ravel())
        self.lines_5.set_xdata([self.r[self.ir] / au, self.r[self.ir] / au])

    def calculate_intensity(self):
        self.k_a = 10.**self.f_kappa(np.log10(self.lam_obs), np.log10(self.A))

        # B_nu has shape (n_wavelength, n_radii)
        # tau and I_nu should have shape (n_wavelength, n_radii, n_azimuth)
        B_nu = az.planck_B_nu(self.nu_obs[:, None], self.T[:, None]).T
        B_nu = np.array(B_nu, ndmin=2)
        tau = (self.sig_d_2D[None, ...] * self.k_a.T[:, None, None, :]).sum(-1)
        self.I_nu = B_nu[:, :, None] * (1 - np.exp(-tau))

    def output(self, fname='modeloutput.txt'):
        with open(fname, 'w') as f:
            f.write(f'# Model with Agas={self.A_gas:.3g}, alpha={self.alpha:.3g}, d2g={self.d2g:.3g}, Sig0={self.sig0:.3g}, a0={self.a0}, p_a={self.pa},  delta phi={self.sigma_y_deg}\n')
            f.write('# Radius  \tSiggas(R)  \tContrastmm \tContrastcm \ta_max\n')
            np.savetxt(f, np.vstack((
                self.r / au,
                self.sig_g,
                self.I_nu[0, ...].max(-1) / self.I_nu[0, ...].min(-1),
                self.I_nu[1, ...].max(-1) / self.I_nu[1, ...].min(-1),
                self.a_max
            )).T, delimiter='\t', fmt='%2.2e')


def main():
    import sys
    _ = Widget(
        fwhm='fwhm' in sys.argv,
        radial_bump='bump' in sys.argv)
    plt.show()


if __name__ == '__main__':
    main()
