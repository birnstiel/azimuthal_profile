import numpy as np
import sys
from io import StringIO

import astropy.constants as c
import astropy.units as u


sig_h2 = 2e-15
mu = 2.3

m_p = c.m_p.cgs.value
k_b = c.k_B.cgs.value


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def make_azim_distri(R, Y, A, T, sig_g_2D, sig_d, alpha, v_frag, rho_s, M_star):
    """
    Creates the 2D distribution of dust assuming a mixing-drift equilibrium at each radius
    for each grain size. The analytical coag-frag size distribution is assumed if sig_d is
    just a 1D array. If sig_d is a 2D grain size distribution, it uses this.

    Arguments:
    ----------
    R  = 1D radial array                                 [cm]
    Y  = azimuthal array where Y[i,j] = R[i]*phi(j)      [cm]
    A  = grain size array                                [cm]
    sig_g_2D = 2D gas surface density map                [g cm^-2]
    sig_d    = either 1d radial grid, then the recipe
               for the coag/frag equilibrium is used
               or a 2D array (A,R) with a distribution   [g cm^-2]
    alpha    = turbulence parameter (scalar or array)    [-]
    v_frag   = fragmentation velocity (scalar or arrry)  [cm s^-1]
    rho_s    = dust internal density                     [g cm^-3]
    M_star   = stellar mass                              [g]
    """
    floor_fact = 1e-50
    if sig_d.ndim == 1:
        recipe = True
    elif sig_d.ndim == 2:
        recipe = False
        distri = sig_d.copy()
        sig_d = sum(sig_d, 0)
    else:
        print('sig_d must be 1 or 2 dimensional!')
        sys.exit(1)
    R = np.array(R, ndmin=1)
    T = np.array(T, ndmin=1)
    A = np.array(A, ndmin=1)
    ny = np.shape(Y)[1]
    nr = len(R)
    na = len(A)
    sig_g = np.average(sig_g_2D, 1)
    alpha = alpha * np.ones(nr)
    v_frag = v_frag * np.ones(nr)

    sig_d_2D = np.zeros([nr, ny, na])
    for ir in np.arange(nr):
        #
        # get the size distribution
        #
        if recipe:
            azim_distri = np.log(A[1] / A[0]) * size_distribution_recipe(A, sig_g=sig_g[ir],
                                                                         sig_d=sig_d[ir], alpha=alpha[ir], T=T[ir], v_f=v_frag[ir], rho_s=rho_s, ret='sig')
        else:
            azim_distri = distri[:, ir]
        azim_max = azim_distri.max()
        #
        # now for each grain size, calculate the equilibrium distribution in phi
        #
        for ia, a in enumerate(A):
            #
            # we neglect all which have a very small mass
            #
            if azim_distri[ia] < floor_fact * azim_max:
                sig_d_2D[ir, :, ia] = azim_distri[ia] / \
                    sum(azim_distri[:]) * np.ones(np.shape(Y)[1])
            else:
                st = a * rho_s / sig_g_2D[ir, :] * np.pi / 2.0
                # old
                #
                # get the steady state drifting/mixing solution
                #
                # sol = solve_drift_diffusion(Y[ir,:],sig_g_2D[ir,:],sig_d[ir]/sig_g[ir],st,alpha)
                #
                # weight each size by the analytical size distribution
                #
                # sol = azim_distri[ia]/sum(azim_distri[:])*sol
                # /old
                # new:
                sol = solve_drift_diffusion_analytical(
                    Y[ir, :], sig_g_2D[ir, :], sig_d[ir] / sig_g[ir] * azim_distri[ia] / sum(azim_distri[:]), st, alpha[ir])
                # /new
                #
                # store it
                #
                sig_d_2D[ir, :, ia] = sol
        progress_bar(100 * (ir + (nr == 1)) / max(1, nr - 1), 'Fitting')
    return sig_d_2D


def solve_drift_diffusion_analytical(xc, rhogas, d2g, st, alpha):
    """
    using the analytical formula

    Arguments:
    ----------
    xc = cell centers
    rhogas = gas density or surface density
    d2g    = dust-to-gas ratio of the dust species
    st     = stokes number as function of x
    alpha  = alpha, either constant or array
    """
    sol = np.exp(-st / alpha + (st / alpha).min()) * rhogas
    sol = sol / np.average(sol) * d2g * np.average(rhogas)
    return sol


def progress_bar(perc, text=''):
    """
    This is a very simple progress bar which displays the given
    percentage of completion on the command line, overwriting its
    previous output.

    Arguments:
    perc    The percentage of completion (float), should be
             between 0 and 100. Only 100.0 finishes with the
             word "Done!".
    text    Possible text for describing the running process.

    Example:
    >>> import time
    >>> for i in linspace(0,100,1000):
    >>>     progress_bar(i,text='Waiting')
    >>>     time.sleep(0.005)
    """
    if text != '':
        text = text + ' ... '
    if perc == 100.0:
        sys.stdout.write('\r' + text + 'Done!\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('\r' + text + '%d %%' % round(perc))
        sys.stdout.flush()


def size_distribution_recipe(a_grid, sig_g=20., sig_d=0.2, alpha=1e-4, T=50., v_f=100., rho_s=1.6, ret='sig'):
    """
    This function returns the result of the grain size distribution recipe
    presented in Birnstiel, Ormel, & Dullemond, A&A (2011), vol. 525, A11

    Arguments:
    a_grid = the grain size grid                      [cm]
    sig_g  = gas surface density                      [g cm^-2]
    sig_d  = dust surface density                     [g cm^-2]
    alpha  = turbulence paramter                      [-]
    T      = mid-plane temperature                    [K]
    v_f    = fragmentation threshold velocity         [cm s^-1]
    rho_s  = dust internal density                    [g cm^-3]
    ret    = either "sig" return sigma                [g cm^-3]
            or      "N"  return number distribution   [cm^-3]
    """
    St = a_grid * rho_s / sig_g * np.pi / 2.
    #
    # gas Reynolds number
    #
    Re = alpha * sig_g * sig_h2 / (2. * mu * m_p)
    #
    # sound speed and gas mean velocity
    #
    cs = np.sqrt(k_b * T / mu / m_p)
    u_gas = cs * np.sqrt(3. / 2. * alpha)
    #
    # constant from Ormel & Cuzzi 2007
    #
    ya = 1.6
    #
    # POINT 1
    # the transition sizes, Eqns. 37, 40, and 27
    #
    a_BT = (8. * sig_g / (np.pi * rho_s) * Re**-0.25 * np.sqrt(mu * m_p / (3 * np.pi * alpha)) * (4. * np.pi / 3. * rho_s)**-0.5)**(2. / 5.)
    a_12 = 1. / ya * 2. * sig_g / (np.pi * rho_s) * Re**-0.5
    a_sett = 2. * alpha * sig_g / (np.pi * rho_s)
    #
    # POINT 2
    # the relative velocities
    #
    du_i_mon = np.zeros(len(a_grid))
    du_i_eq = np.zeros(len(a_grid))
    for i, a in enumerate(a_grid):
        eps = (a - a_12) / (4. * a_12)
        #
        # with monomers
        #
        if a < a_12:
            du_i_mon[i] = u_gas * Re**0.25 * (St[i] - St[0])
        elif a_12 < a < 5. * a_12:
            du_i_mon[i] = (1. - eps) * u_gas * Re**0.25 * \
                (St[i] - St[0]) + eps * u_gas * np.sqrt(3. * St[i])
        elif a >= 5 * a_12:
            du_i_mon[i] = u_gas * np.sqrt(3. * St[i])
        #
        # with equal sized particles
        #
        if a < a_12:
            du_i_eq[i] = 0.0
        elif a >= a_12:
            du_i_eq[i] = np.sqrt(2. / 3.) * du_i_mon[i]
    #
    # POINT 3
    # now np.pi the fragmentation limits
    #
    del_v = 0.2 * v_f
    i_a_L = find(du_i_mon >= v_f - del_v)[0]
    i_a_P = find(du_i_eq >= v_f - del_v)[0]
    i_a_R = find(du_i_eq >= v_f)[0]
    a_L = a_grid[i_a_L]
    a_P = a_grid[i_a_P]
    a_R = a_grid[i_a_R]
    #
    # POINT 4
    #
    V = cs * (8. * mu * m_p * sig_g / (alpha * sig_h2))**0.25 * \
        np.sqrt(3. / 4. * alpha / (sig_g * ya))
    J = (2.5**-9 + (1.1**9 + (1. + 2. * np.sqrt(3.) * V / v_f)**9)**-1)**(-1. / 9.)
    #
    # POINT 5
    #
    f = np.zeros(len(a_grid))
    for i, a in enumerate(a_grid):
        if i == 0:
            f[i] = 1
        else:
            #
            # select slope
            #
            if a >= a_sett:
                settling = True
            else:
                settling = False
            if a <= a_BT:
                if settling:
                    slope = 5. / 4.
                else:
                    slope = 3. / 2.
            elif a <= a_12:
                if settling:
                    slope = 0.0
                else:
                    slope = 1. / 4.
            else:
                if settling:
                    slope = 1. / 4.
                else:
                    slope = 1. / 2.
            #
            # fill bin
            #
            if a < a_P:
                f[i] = f[i - 1] * (a / a_grid[i - 1])**slope
    #
    # include the "jump"
    #
    i_12 = find(a_grid >= a_12)[0]
    f[i_12:] = f[i_12:] / J
    #
    # POINT 6
    #
    a_inc = 0.3 * a_P
    for i, a in enumerate(a_grid):
        if a_inc < a < a_P:
            f[i] = f[i] * (2. - (a - a_P) / (a_inc - a_P))
    #
    # POINT 7
    #
    sig = max(min(abs(a_R - a_P), abs(a_L - a_P)) / np.sqrt(np.log(2)), 0.1 * a_P)
    b = np.zeros(len(a_grid))
    for i, a in enumerate(a_grid):
        b[i] = 2. * f[i_a_L] * np.exp(-(a - a_P)**2 / sig**2)
    #
    # POINT 8
    #
    F = np.zeros(len(a_grid))
    for i, a in enumerate(a_grid):
        if a <= a_L:
            F[i] = f[i]
        elif a_L <= a < a_P:
            F[i] = max(f[i], b[i])
        elif a_P <= a <= a_R:
            F[i] = b[i]
        else:
            F[i] = 1e-300
    #
    # POINT 9
    # normalize to the proper dust-to-gas ratio
    #
    if ret == 'sig':
        F = F / sum(F) * sig_d / np.log(a_grid[1] / a_grid[0])
        return F
    elif ret == 'N':
        N = sig_d * F / (4. * np.pi / 3. * rho_s * a**4) / np.trapz(F, x=np.log(a_grid))
        return N
    else:
        print('ERROR: the output specification needs to be \'sig\' or \'N\'')


def planck_B_nu(freq, T):
    """
    Calculates the value of the Planck-Spectrum
    B(nu,T) of a given frequency nu and temperature T

    Arguments
    ---------
    nu : float or array
        frequency in 1/s or with astropy.units

    T: float
        temperature in K or in astropy.units

    Returns:
    --------
    B : float
        value of the Planck-Spectrum at frequency nu and temperature T
        units are using astropy.units if the input values use those, otherwise
        cgs units: erg/(s*sr*cm**2*Hz)

    """
    if isinstance(T, u.quantity.Quantity):
        use_units = True
    else:
        T = T * u.K
        use_units = False

    if not isinstance(freq, u.quantity.Quantity):
        freq *= u.Hz

    T = np.array(T.value, ndmin=1) * T.unit
    freq = np.array(freq.value, ndmin=1) * freq.unit

    f_ov_T = freq[np.newaxis, :] / T[:, np.newaxis]
    mx = np.floor(np.log(np.finfo(f_ov_T.ravel()[0].value).max))
    exp = np.minimum(f_ov_T * c.h / c.k_B, mx)
    exp = np.maximum(exp, -mx)

    output = 2 * c.h * freq**3 / c.c**2 / (np.exp(exp) - 1.0) / u.sr

    cgsunit = 'erg/(s*sr*cm**2*Hz)'
    if use_units:
        return output.to(cgsunit).squeeze()
    else:
        return output.to(cgsunit).value.squeeze()


class Capturing(list):
    """Context manager capturing standard output of whatever is called in it.

    Examples
    --------
    >>> with Capturing() as output:
    >>>     do_something(my_object)

    `output` is now a list containing the lines printed by the function call.

    This can also be concatenated

    >>> with Capturing() as output:
    >>>    print 'hello world'
    >>>
    >>> print('displays on screen')
    >>>
    >>> with Capturing(output) as output:
    >>>     print('hello world2')
    >>>
    >>> print('done')
    >>> print('output:', output)
    done
    output: ['hello world', 'hello world2']

    Copied from [this stackoverflow answer](http://stackoverflow.com/a/16571630/2108771)

    """

    def __enter__(self):
        """Start capturing output when entering the context"""
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """Get & return the collected output when exiting context"""
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def is_interactive():
    """
    Function to test if this code is executed interactively (in notebook or
    console) or if it is run as a script.

    Returns:
    --------
    True if run in notebook or console, False if script.

    """
    import __main__ as main
    return not hasattr(main, '__file__')
