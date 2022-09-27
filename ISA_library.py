#v2.1 - Andre Celere
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

#conversion constants
m2ft = 3.28084 #1 meter to feet
ft2m = 1/m2ft
kt2ms = 0.51444444 #knots to meters per second
ms2kt = 1/kt2ms
RPM2rads = 1/60*2*math.pi
C2K = 273.15 #add to go from C to K
kgm32slug = 0.00237717/1.225 #

#constants definitions for ISA Atmosphere
TROPOSPHERE = 36089.24  #feet
T0 = 288.15 #Kelvin
p0 = 101325 #Pa
L_m = -6.5/1000 #K/m
L = L_m/m2ft #K/ft
a0 = 340.3 #m/s

STRATOSPHERE = 65617 #feet
Ts = 216.5 #Kelvin - temp at stratosphere
ps = 22632.06 #Pa


rho0 = 1.225 #kg/m3
R = 287.053 #m2/s2/K
ag_zero = -5.25588 *  R * L #in m/s2 -> gravity acceleration
gamma = 1.4 #adiabatic coefficient for air

#This function returns a pressure given an altitude
def getPressure(h):
    #h in feet
    #p in Pascals
    if h <= TROPOSPHERE:
        p = p0*(1+L/T0*h)**(-ag_zero/(R*L))
    else:
        p = ps*math.exp(-1*(h-TROPOSPHERE)/(R*Ts/ag_zero))
    return p

# temperature ratio
def theta(h):
    #h in feet
    if h <= TROPOSPHERE:
        theta_calc = 1+(L/T0)*h
    else:
        theta_calc = Ts/T0
    return theta_calc

#this function returns the temperature ratio given OAT
def thetaOAT(OAT):
    #OAT in C
    return (OAT+C2K)/T0

# pressure ratio
def delta(h):
    #h in feet
    delta_calc = getPressure(h)/p0
    return delta_calc

#this function returns the altitude given a pressure ratio (delta)
def inv_delta(delta):
    #return the altitude for that delta
    idelta = (delta**(1/5.255863)-1)/(-0.00000687535)
    return idelta

#density ratio
def sigma(h):
    #h in feet
    sigma_calc = delta(h)/theta(h)
    return sigma_calc

#this function returns the altitude given a density ratio (sigma)
def inv_sigma(sigma):
    isigma = ((sigma**0.235)-1)/(-0.00000687535)
    return isigma

#this function returns an altitude given a pressure
def getAltitude(p):
    #p in Pascals
    #h in feet
    if p >= ps:
        h = T0/L*((p/p0)**(-(R*L/ag_zero))-1)
    else:
        h = TROPOSPHERE+R*Ts/ag_zero*math.log(p/ps)
    return h

#this function returns the speed of sound given an absolute temperature
def aSpdSound(T):
    #T comes in Kelvin
    if T >= 0:
        return math.sqrt(gamma*R*T)
    else:
        return 0
    
#this function returns the density altitude given an altitude and a temperature
def dAltitude(h, t):
    temp_constant = 0.00000687535
    DA = (((((1-temp_constant*h)**5.2561)/((t+C2K)/T0))**0.235)-1)/(-temp_constant)
    return DA

#this function returns the OAT for ISA atmosphere, given and altitude
def getOAT_ISA(h):
    #given height, what is the ISA OAT?
    OAT = ((T0*(1-0.00000687535*h)))-C2K
    #outouts in C
    return OAT

#statistic function to calculate coefficient of determination (R squared)
#inputs are a fitted function and x,y original vectors
def get_r(fitted_fn, x, y):
    yhat = fitted_fn(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y-ybar)**2)
    r_line = ssreg/sstot
    return r_line

#these are the vectorized function forms for speed
vtheta = np.vectorize(theta)
vdelta = np.vectorize(delta)
vsigma = np.vectorize(sigma)


#this function returns Mach number for KTAS and Hp
def getMach(KTAS, Hp):
    #KTAS speed in knots
    #Hp altitude in feet
    return((KTAS)/(a0*ms2kt*theta(Hp)))

#this function returns KEAS from KCAS and Hp
def Vc2Ve(KCAS, Hp):
    current_delta = delta(Hp)
    p1 = 1 + 0.2*(KCAS/(a0*ms2kt))**2
    p2 = p1**(3.5)
    p3 = (p2-1)/current_delta
    p4 = (p3+1)**(1/3.5)
    p5 = np.sqrt((p4-1)*(current_delta*(a0*ms2kt)**2)/(0.2))
    return p5

#this function returns KTAS from KEAS and Hp
def Ve2Vt(KEAS, Hp):
    current_sigma = sigma(Hp)
    return KEAS/np.sqrt(current_sigma)

#wrapper for directly going from KCAS to KTAS
def Vc2Vt(KCAS, Hp):
    KEAS = Vc2Ve(KCAS, Hp)
    return Ve2Vt(KEAS, Hp)

#this function returns KCAS from KEAS and Hp
def Ve2Vc(KEAS, Hp):
    current_delta = delta(Hp)
    p1 = 1 + (0.2/current_delta)*(KEAS/(a0*ms2kt))**2
    p2 = p1**(3.5)
    p3 = (p2-1)*current_delta
    p4 = (p3+1)**(1/3.5)
    p5 = np.sqrt((p4-1)*(current_delta*(a0*ms2kt)**2)/(0.2))
    return p5

#this function returns KEAS from KTAS and Hp
def Vt2Ve(KTAS, Hp):
    current_sigma = sigma(Hp)
    return KTAS*np.sqrt(current_sigma)

#wrapper for directly going from KTAS to KCAS
def Vt2Vc(KTAS, Hp):
    KEAS = Vt2Ve(KTAS, Hp)
    return Ve2Vc(KEAS, Hp)

#this function returns KTAS number for Mach and Hp
def M2Vt(M, Hp):
    #Mach speed in knots
    #Hp altitude in feet
    #returns true airspeed in kts
    return M*a0*ms2kt*theta(Hp)

#this function returns KTAS number for Mach and Hp
def Vc2M(KCAS, Hp):
    #Calibrated speed in knots
    #Hp altitude in feet
    #returns Mach
    return getMach(Vc2Vt(KCAS, Hp), Hp)

#this function returns the total pressure for a given speed and altitude, considering compressibility
def PTot(KTAS, Hp):
    #speed in kts
    #alt in feet
    #returns pressure in pascals
    p = getPressure(Hp)
    u = KTAS*kt2ms
    calc_sigma = sigma(Hp)
    rho = rho0 * calc_sigma
    M = getMach(KTAS, Hp)
    series = 1 + (M**2)/4 + (M**4)/40 + (M**6)/240
    return p + 0.5 * rho * u**2 * series

#this function returns the total temperature for a given speed and altitude, considering ISA
def TAT(KTAS, Hp, k):
    #speed in kts
    #alt in feet
    #returns temperature in K
    M = getMach(KTAS, Hp)
    T = T0 * theta(Hp)
    return T * (1 + 0.2*k*M**2)

#this function returns KTAS from KEAS and Hp, considering deltaISA
def Ve2Vt_dISA(KEAS, Hp, deltaISA):
    current_delta = delta(Hp)
    current_OAT = getOAT_ISA(Hp) + deltaISA + C2K
    current_theta = current_OAT / T0
    current_sigma = current_delta / current_theta
    return KEAS/np.sqrt(current_sigma)

#wrapper for directly going from KCAS to KTAS
def Vc2Vt_dISA(KCAS, Hp, deltaISA):
    KEAS = Vc2Ve(KCAS, Hp)
    return Ve2Vt_dISA(KEAS, Hp, deltaISA)

#calculate indicated airspeed from dynamic pressure
def getVc(pd):
    #pd in Pascals
    # Vc in KIAS
    Vc = np.sqrt((2 * gamma)/(gamma - 1) * (p0) / (rho0) * (((abs(pd) / p0) + 1)**((gamma - 1) / gamma) - 1)) * ms2kt
    return Vc