import numpy as np
from plot_signals import plot_signals
from scipy.integrate import odeint
import scipy

def calc_synaptic_currents(gsynE, gsynI, EsynE, EsynI, vhalf, kv, V, b, c, d):
    tonic_drives = np.dot(c.transpose(),d).transpose() # should be a row
    I_tonicE = gsynE*(V - EsynE)*tonic_drives
    pos = np.maximum(b, 0)
    neg = np.maximum(-b, 0)
    IsynE = I_tonicE + gsynE*np.array([(V - EsynE)])*np.dot(fun(V, vhalf, kv),pos)
    IsynI = gsynI*np.array([V - EsynI])*np.dot(fun(V, vhalf, kv),neg)
    return IsynE.squeeze(), IsynI.squeeze()

def fun(v, vhalf, kv):
    return 1.0 / (1.0 + np.exp(-(v - vhalf) / kv))


def mnap(v1):
    return 1.0 / (1.0 + np.exp(-(v1 + 40.0) / 6.0))


def hinfnap(v1):
    return 1.0 / (1.0 + np.exp((v1 + 48.0) / 6.0))


def mk(v1):
    return 1.0 / (1.0 + np.exp(-(v1 + 29.0) / 4.0))


def taonap(v1, tnapmax):
    return tnapmax / np.cosh((v1 + 48.0) / 12.0)

def I(t,t1,t2):
    if (t < t1) or (t > t2):
        return 0
    else:
        return 600

def vectorfield(w, t, p):
    num_nrns = int(len(w)/2)
    V = w[:num_nrns]
    M = w[num_nrns:]
    Capacity, gl, gnap, gk, gad, gsynE, gsynI, Ena, Ek, El, EsynE, EsynI, vhalf, kv, tnapmax, tad, kad, b, c, d = p

    # unique properties of bursting population
    Inap = gnap * mnap(V[0]) * M[0] * (V[0] - Ena)
    Ik = gk * ((mk(V[0])) ** 4) * (V[0] - Ek)

    # common properties
    Il = gl*(V - El) #make sure it's a vector
    Iad = gad * M * (V - Ek) # 0 th element will not be used anywhere
    IsynE , IsynI = calc_synaptic_currents(gsynE, gsynI, EsynE, EsynI, vhalf, kv, V, b, c, d)

    f = []

    f.append((-Inap - Ik - Il[0] - IsynE[0] - IsynI[0]) / Capacity)
    for i in range(1,num_nrns):
        if i != 5:
            f.append((-Iad[i] - Il[i] - IsynE[i] - IsynI[i]) / Capacity)
        else:
            f.append((-Iad[i] - Il[i] - IsynE[i] - IsynI[i] + 0.85*I(t, 25000, 35000)) / Capacity)

    f.append((hinfnap(V[0]) - M[0]) / taonap(V[0], tnapmax))
    for i in range(1,num_nrns):
        f.append((kad[i]*fun(V[i], vhalf, kv) - M[i]) / tad[i])

    return f


def model(b, c, vectorfield, stoptime):
    num_nrns = b.shape[0]
    num_drives = 3
    # Parameter values
    Capacity = 20
    gnap = 5.0
    gk = 5.0
    gad = 10.0
    gl = 2.8
    gsynE = 10
    gsynI = 60
    Ena = 50
    Ek = -85
    El = -60
    EsynE = 0
    EsynI = -75
    vhalf = -30
    kv = 4
    tad = 2000*np.ones(num_nrns)
    tad[0] = 0.0
    kad = 0.9*np.ones(num_nrns)
    kad[0] = 0.0
    tnapmax = 6000
    d = np.ones(num_drives)
    # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6
    numpoints = 8192*2
    # Create the time samples for the output of the ODE solver.
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

    # Pack up the parameters and initial conditions:
    p = [Capacity, gl, gnap, gk, gad, gsynE, gsynI, Ena, Ek, El, EsynE, EsynI, vhalf, kv, tnapmax, tad, kad, b, c, d]
    np.random.seed(0)
    V0 = -70 + 40 * np.random.rand(num_nrns)
    M0 = np.random.rand(num_nrns)
    w0 = []
    for i in range(num_nrns):
        w0.append(V0[i])
    for i in range(num_nrns):
        w0.append(M0[i])

    # Call the ODE solver.
    wsol = odeint(vectorfield, w0, t, args=(p,))

    S = [[] for i in range(num_nrns)]
    res = []
    res.append(np.array(t))
    for w in wsol:
        for i in range(num_nrns):
            S[i].append(fun(w[i],vhalf,kv))
    S = [np.array(x) for x in S]

    res = []

    res.append(t)
    for i in range(num_nrns):
        res.append(S[i])
    return res

