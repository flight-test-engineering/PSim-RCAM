import numpy as np

'''
module to calculate ISA atmosphere and speed conversions
change log:
v0: initial release, ISA only
v1: added speed conversions
'''


# constants
R_star = 8.31432 * 1E-3 # N*m/(kmol K) -> ISA page 2
Mol_W_0 = 28.9644 #kg/kmol -> ISA page 9
R = 287.053 # R_star/Mol_W_0 [m2/s2/K]
gamma = 1.4 # for air -> ISA page 4
g_SL = 9.80665 # [m/s2] -> ISA page 2
m2ft = 3.28084
ft2m = 1 / m2ft
kt2ms = 0.5144
ms2kt = 1 / kt2ms

# define strata

# troposphere
L = -6.5 / 1000 #K/m
Hc_t_tropo = 36089.24 # [ft]
T_SL = 288.15 # [K]
p_SL = 101325 # [Pa]
rho_SL = 1.225 # [kg/m3]
a_SL = np.sqrt(gamma * R * T_SL) # speed of sound at sea level [m/s]

# stratosphere
Hc_b_strato = Hc_t_tropo + 0.01 # [ft]
Hc_t_strato = 65616.8 # [ft]
T_b_strato = 216.65 # [K]
p_b_strato = 22632.06 # [Pa]
p_t_strato = 5474.88 # [Pa]
delta_b_strato = p_b_strato / p_SL
delta_t_strato = p_t_strato / p_SL
theta_b_strato = T_b_strato / T_SL
rho_b_strato = 0.36392 # [kg/m3]
rho_t_strato = 0.08803 #[kg/m3]
sigma_b_strato = rho_b_strato / rho_SL
sigma_t_strato = rho_t_strato / rho_SL

def delta_non_v(Hc:float)->float:
    '''
    this function calculates 'delta', the ISA pressure ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'delta'
    '''
   
    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))**(-g_SL / (L * R))
    elif Hc <= Hc_t_strato:
        return delta_b_strato * np.exp(-(g_SL / (R * T_b_strato))*((Hc - Hc_b_strato)*ft2m))
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")

delta = np.vectorize(delta_non_v)

        
def p(Hc:float)->float:
    '''
    this function calculates the ISA pressure, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        p: in Pascals
    '''

    return delta(Hc) * p_SL

def theta_non_v(Hc:float)->float:
    '''
    this function calculates 'theta', the ISA temperature ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'theta'
    
    '''

    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))
    elif Hc <= Hc_t_strato:
        return theta_b_strato
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")
        
theta = np.vectorize(theta_non_v)

def T(Hc:float)->float:
    '''
    this function calculates the ISA temperature, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        T: in Kelvin
    
    '''

    return theta(Hc) * T_SL

def sigma_non_v(Hc:float)->float:
    '''
    this function calculates 'sigma', the ISA density ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'sigma'
    
    '''
    
    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))**(-g_SL / (L * R) - 1)
    elif Hc <= Hc_t_strato:
        return sigma_b_strato * np.exp(-(g_SL / (R * T_b_strato))*((Hc - Hc_b_strato)*ft2m))
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")
        
sigma = np.vectorize(sigma_non_v)

def rho(Hc:float)->float:
    '''
    this function calculates the ISA density, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        rho: in kg/m3
    
    '''

    return sigma(Hc) * rho_SL

def inv_delta_non_v(delta:float)->float:
    '''
    this function calculates ISA pressure altitude for a given pressure ratio 'delta'
    limited to top of stratosphere
    inputs:
        delta  [non-dimensional]
    outputs:
        Hc: in feet
        
    
    '''
    
    if delta > delta_b_strato:
        return (T_SL / L) * ((delta)**(-(L * R) / g_SL) - 1) * m2ft
    elif delta >= delta_t_strato:
        return ((((-R * T_b_strato) / g_SL) / ft2m) * np.log(((delta) * np.exp((-g_SL / (R * T_b_strato)) * Hc_b_strato * ft2m)) / (p_b_strato / p_SL)))
    else:
        raise ValueError("Pressure/delta lower than stratospheric limit - outside bounds for this function")

inv_delta = np.vectorize(inv_delta_non_v)

def inv_p(p:float)->float:
    '''
    this function calculates the ISA pressure altitude, for a given pressure
    limited to top of stratosphere
    inputs:
        p: in Pascals
    outputs:
        Hc: in feet
    
    '''

    return inv_delta(p / p_SL)

def inv_sigma_non_v(sigma:float)->float:
    '''
    this function calculates ISA pressure altitude for a given density ratio
    limited to top of stratosphere
    inputs:
        sigma [non-dimensional]
    outputs:
        Hc: in feet
        
    
    '''
    
    if sigma > sigma_b_strato:
        return (T_SL / L)*(((sigma)**(1 / (-g_SL / (L * R) - 1)) - 1))*m2ft #validado
    elif sigma >= sigma_t_strato:
        return ((((R * T_b_strato) / -g_SL) / ft2m) * np.log(((sigma) * np.exp((-g_SL / (R * T_b_strato)) * Hc_b_strato * ft2m)) / (rho_b_strato / rho_SL))) #validado
    else:
        raise ValueError("Density/sigma below stratospheric limit - outside bounds for this function")

inv_sigma = np.vectorize(inv_sigma_non_v)

def inv_rho(rho:float)->float:
    '''
    this function calculates the ISA pressure altitude, for a given density
    limited to top of stratosphere
    inputs:
        rho: in kg/m3
    outputs:
        Hc: in feet
    
    '''

    return inv_sigma(rho / rho_SL)

def Vc2M(Vc, Hc):
    '''
    this function calculates Mach number for a given calibrated airspeed and altitude
    inputs:
        Vc: calibrated airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Mach number
    '''
    d = delta(Hc)
    M = np.sqrt(5 * (((1 / d) * ((1 + 0.2 * ((Vc * kt2ms) / a_SL)**2)**(7/2) - 1) + 1)**(2/7) - 1))
    return M

def M2Vc(M, Hc):
    '''
    this function calculates calibrated airspeed from a given Mach and altitude
    inputs:
        Mach number
        Hc: calibrated altitude in ft
    outputs:
        Vc: calibrated airspeed in kts
    '''
    d = delta(Hc)
    Vc = (a_SL * np.sqrt(5 * (((d * ((1 + 0.2 * M**2)**(7/2) - 1)) + 1)**(2/7) - 1))) * ms2kt
    return Vc

def Vt2Ve(Vt, Hc):
    '''
    this function calculates equivalent airspeed from a given true and altitude
    inputs:
        Vt: true airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Ve: equivalent airspeed in kts
    '''
    s = sigma(Hc)
    return Vt * np.sqrt(s)

def Ve2Vt(Ve, Hc):
    '''
    this function calculates true airspeed from a given equivalent and altitude
    inputs:
        Ve: equivalent airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Vt: true airspeed in kts
    '''
    s = sigma(Hc)
    return Ve / np.sqrt(s)

def M2Ve(M, Hc):
    '''
    this function calculates equivalent airspeed from a given Mach and altitude
    inputs:
        Mach number
        Hc: calibrated altitude in ft
    outputs:
        Vc: calibrated airspeed in kts
    '''
    Pt_Pa_over_Pa = (1 + 0.2 * M**2)**(7/2) - 1
    Pa = p(Hc)
    Ve = np.sqrt((1 / rho_SL) * (7 * Pa * ((Pt_Pa_over_Pa + 1)**(2/7) - 1)))
    return Ve * ms2kt

def M2Vt(M, Hc):
    '''
    this function calculates true airspeed from a given Mach and altitude
    inputs:
        Mach number
        Hc: calibrated altitude in ft
    outputs:
        Vc: calibrated airspeed in kts
    '''
    Pt_Pa_over_Pa = (1 + 0.2 * M**2)**(7/2) - 1
    Pa = p(Hc)
    Ve = np.sqrt((1 / rho_SL) * (7 * Pa * ((Pt_Pa_over_Pa + 1)**(2/7) - 1)))
    return Ve2Vt(Ve * ms2kt, Hc)

def Vt2M(Vt, Hc):
    '''
    this function calculates Mach from a given true and altitude
    inputs:
        Vt: true airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        M: Mach number
    '''
    return Vt / (np.sqrt(gamma * R * T(Hc)) * ms2kt)

def Vt2Vc(Vt, Hc):
    '''
    this function calculates calibrated airspeed from a given true and altitude
    inputs:
        Vt: true airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Vc: calibrated airspeed in kts
    '''
    M = Vt2M(Vt, Hc)
    Vc = M2Vc(M, Hc)
    return Vc

def Vc2Vt(Vc, Hc):
    '''
    this function calculates true airspeed from a given calibrated and altitude
    inputs:
        Vc: calibrated airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Vt: true airspeed in kts
    '''
    M = Vc2M(Vc, Hc)
    Vt = M2Vt(M, Hc)
    return Vt

def Vc2Ve(Vc, Hc):
    '''
    this function calculates equivalent airspeed from a given calibrated and altitude
    inputs:
        Vc: calibrated airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Ve: equivalent airspeed in kts
    '''
    M = Vc2M(Vc, Hc)
    Ve = M2Ve(M, Hc)
    return Ve

def Ve2Vc(Ve, Hc):
    '''
    this function calculates calibrated airspeed from a given equivalent and altitude
    inputs:
        Ve: equivalent airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        Vc: calibrated airspeed in kts
    '''
    Vt = Ve2Vt(Ve, Hc)
    M = Vt2M(Vt, Hc)
    Vc = M2Vc(M, Hc)
    return Vc

def Ve2M(Ve, Hc):
    '''
    this function calculates Mach number from a given equivalent and altitude
    inputs:
        Ve: equivalent airspeed in kts
        Hc: calibrated altitude in ft
    outputs:
        M: Mach number
    '''
    Vt = Ve2Vt(Ve, Hc)
    M = Vt2M(Vt, Hc)
    return M