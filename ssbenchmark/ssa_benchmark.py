# -*- coding: utf-8 -*-
'''
Created on Jul 10, 2017

@author: arosete
'''
# This program simulates spectrum sensing to produce receiver operating characteristic (ROC)
# curves for some given spectrum sensing algorithm (SSA), as well as the area under the curve
# (AUC) versus number of samples measure of performance.

# This code is written by 
# Andre Rosete <andre.rosete@colorado.edu; andre.rosete@nist.gov>,
# Ph.D. Student at the University of Colorado Boulder and
# PREP Associate at the National Institute of Standards and Technology,
# Boulder, CO, USA

# This code utilizes the TracyWidom.py tool written by 
# Yao-Yuan Mao
# Postdoctoral Fellow at the University of Pittsburgh, Pennsylvania, USA
# https://gist.github.com/yymao/7282002
# The tool is used to perform calculations related to the Tracy-Widom distribution
# which is utilized in the threshold-setting functions of some SSAs



# Use Numpy for mathematics
import numpy
# Use Scipy for Linear Algebra
import scipy.linalg
# Use Scipy for the Inverse Q Function
import scipy.special
# Use Scipy to rank data
import scipy.stats
# Use Numba JIT (just-in-time) compiler to make faster C code
# from numba import jit
# Use matplotlib pyplot to plot data
import matplotlib.pyplot as plt
# Need TracyWidom.py by Yao-Yuan Mao (https://gist.github.com/yymao) to compute the Tracy-Widom CDF, used in the MME method
import TracyWidom

# Get PI = 3.141592653589793
PI = numpy.pi

def SampleAutoCorrelation(x, l):
    """
    SampleAutoCorrelation builds a list of eigenvalues (lambdas) representing the sample auto-correlation for a sliding covariance window position.
    Used to assist the creation of a statistical covariance matrix R.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 26
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param l: current sliding covariance window position (integer)
    
    Returns:
    :return sample auto-correlation (float32)
    """
    sum_terms = []
    Ns = len(x)
    if(Ns > 0):
        for m in range(0, Ns):
            sum_terms.append(x[m] * x[m - l])
        return((1 / Ns) * sum(sum_terms))
    else:
        return(0)
    
def StatCovarianceMatrix(x, L):
    """
    StatCovarianceMatrix builds the statistical covariance matrix R for a given set of received signal samples and a covariance window size.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 27
    
    Parameters:
    :param x: received signal samples (float32 list)
    
    Returns:
    :return statistical covariance matrix R (float32 matrix)
    """
    
    lambdas = []
    for l in range(0, L):
        lambdas.append(SampleAutoCorrelation(x, l))
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 26
    temp0 = scipy.linalg.toeplitz(lambdas)
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 21
    temp1 = numpy.var(x) * numpy.identity(len(temp0))
    return(temp0 + temp1)

def EnergyDetection(x):
    """
    EnergyDetection calculates the aggregate energy in the provided samples.
    
    Parameters:
    :param x: received signal samples (float32 list)
    
    Returns:
    :return aggregate energy magnitude (float32)
    """
    return((1 / len(x)) * (sum(abs(x) ** 2)))

def EnergyThreshold(x, pfa):
    """
    EnergyThreshold generates an energy detection threshold for a given signal.
    
    Parameters:
    :param x: received signal samples (float32 list)
    
    Returns:
    :return energy magnitude threshold (float32)
    """
    return((InverseQFunction(pfa) / numpy.sqrt(len(x)) + 1))

def CovarianceAbsoluteValueDetection(x, L):
    """
    CovarianceAbsoluteValueDetection calculates the spectrum sensing test statistic for a given signal using the Covariance Absolute Value (CAV) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Algorithm 1
    
    Parameters:
    :param x: received signal samples (float32 list)
    
    Returns:
    :return test statistic (float32)
    """
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 27
    Rx = StatCovarianceMatrix(x, L)
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 29
    T1 = (1 / L) * numpy.sum(abs(Rx))
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 30
    T2 = (1 / L) * numpy.matrix.trace(abs(Rx))
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Algorithm 1, Step 6
    return(T1 / T2)

def CovarianceFrobeniusNormDetection(x, L):
    """
    CovarianceFrobeniusNormDetection calculates the spectrum sensing test statistic for a given signal using the Covariance Frobenius Norm (CFN) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Algorithm 2
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return test statistic (float32)
    """
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 27
    Rx = StatCovarianceMatrix(x, L)
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 31
    T1 = (1 / L) * numpy.sum((abs(Rx)) ** 2)
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 32
    T2 = (1 / L) * numpy.matrix.trace((abs(Rx) ** 2))
    # Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Algorithm 2, Step 6
    return(T1 / T2)

# Calculate the Inverse Q Function
# @jit
def InverseQFunction(p):
    """
    InverseQFunction finds the value has the given probability of being along the normal/Gaussian distribution.
    
    Parameters:
    :param p: probability (float32, [0, 1])
    
    Returns:
    :return value satisfying the given probability along a Gaussian distribution (float32)
    """
    return(1.41421356237 * scipy.special.erfcinv(2 * p))

# Calculate Absolute Value Threshold
# Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 53
# @jit
def CovarianceAbsoluteValueThreshold(x, L, pfa):
    """
    CovarianceAbsoluteValueThreshold calculates the spectrum sensing threshold for a given signal using the Covariance Absolute Value (CAV) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 53
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return threshold (float32)
    """
    if(len(x) > 0):
        return((1 + (L - 1) * numpy.sqrt(2 / (len(x) * PI))) / (1 + InverseQFunction(1 - pfa) * numpy.sqrt(2 / len(x))))
    else:
        return(0)

def CovarianceFrobeniusNormThreshold(x, L, pfa):
    """
    CovarianceFrobeniusNormThreshold calculates the spectrum sensing threshold for a given signal using the Covariance Frobenius Norm (CFN) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Covariance-Based Signal Detections for Cognitive Radio, Equation 54
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return threshold (float32)
    """
    return((L + len(x) + 1) / (len(x) + 2 + (InverseQFunction(1 - pfa) * numpy.sqrt(8 * len(x) + 40 + (48 / len(x))))))

# Maximum-Minimum Eigenvalue (MME) Detection
# Yonghong Zeng and Ying-Chang Liang, Maximum-Minimum Eigenvalue Detection for Cognitive Radio, Algorithm 1
def MaximumMinimumEigenvalueDetection(x, L):
    """
    MaximumMinimumEigenvalueDetection calculates the spectrum sensing test statistic for a given signal using the Maximum-Minimum Eigenvalue (MME) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Maximum-Minimum Eigenvalue Detection for Cognitive Radio, Algorithm 1
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return test statistic (float32)
    """
    Rx = StatCovarianceMatrix(x, L)
    eigenvalues = numpy.linalg.eigvalsh(Rx)
    T1 = max(eigenvalues)
    T2 = min(eigenvalues)
    return(T1 / T2)

# Maximum-Minimum Eigenvalue (MME) Threshold
# Yonghong Zeng and Ying-Chang Liang, Maximum-Minimum Eigenvalue Detection for Cognitive Radio, Equation 20
def MaximumMinimumEigenvalueThreshold(x, L, pfa):
    """
    MaximumMinimumEigenvalueThreshold calculates the spectrum sensing threshold for a given signal using the Maximum-Minimum Eigenvalue (MME) algorithm.
    Yonghong Zeng and Ying-Chang Liang, Maximum-Minimum Eigenvalue Detection for Cognitive Radio, Equation 20
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return threshold (float32)
    """
    Ns = len(x)
    M = 1  # Number of receiving antennas
    temp1 = (numpy.sqrt(Ns) + numpy.sqrt(M * L)) ** 2
    temp2 = (numpy.sqrt(Ns) - numpy.sqrt(M * L)) ** 2
    temp3 = (numpy.sqrt(Ns) + numpy.sqrt(M * L)) ** (-2 / 3)
    temp4 = (Ns * M * L) ** (1 / 6)
    TW = TracyWidom.TracyWidom(beta=2)
    inv_F = TW.cdfinv(1.0 - pfa)
    return((temp1 / temp2) * (1 + (temp3 / temp4) * inv_F))

# Arithmetic-to-Geometric Mean (AGM) Detection
def ArithmeticToGeometricMeanDetection(x, L):
    """ 
    ArithmeticToGeometricMeanDetection calculates the spectrum sensing test statistic for a given signal using the Arithmetic-to-Geometric Mean (AGM) algorithm.
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return test statistic (float32)
    """
    Rx = StatCovarianceMatrix(x, L)
    eigenvalues = numpy.linalg.eigvalsh(Rx)
    Ns = len(x)
    return(((1 / Ns) * sum(eigenvalues)) / ((numpy.matrix.cumprod(eigenvalues) ** (1 / Ns))))
           
# Energy with Minimum Eigenvalue (EME) Detection
# Yonghong Zeng and Ying-Chang Liang, Eigenvalue-based spectrum sensing algorithms for cognitive radio, Algorithm 2
def EnergyWithMinimumEigenvalueDetection(x, L):
    """
    EnergyWithMinimumEigenvalueDetection computes the spectrum sensing test statistic according to
    Yonghong Zeng and Ying-Chang Liang, Eigenvalue-based spectrum sensing algorithms for cognitive radio, Algorithm 2
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return test statistic (float32)
    """
    Rx = StatCovarianceMatrix(x, L)
    eigenvalues = numpy.linalg.eigvalsh(Rx)
    return(EnergyDetection(x) / min(eigenvalues))
    
def EnergyWithMinimumEigenvalueThreshold(x, L, pfa):
    """ 
    EnergyWithMinimumEigenvalueThreshold computes the spectrum sensing detection threshold according to
    Yonghong Zeng and Ying-Chang Liang, Eigenvalue-based spectrum sensing algorithms for cognitive radio, Equation 34
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return threshold (float32)
    """
    Ns = len(x)
    M = 1  # Number of antennas
    return((numpy.sqrt(2 / (M * Ns)) * InverseQFunction(pfa) + 1) * (Ns / (numpy.sqrt(Ns) - numpy.sqrt(M * L)) ** 2))

def MaximumEigenvalueTraceDetection(x, L):
    """
    MaximumEigenvalueTraceDetection computes the spectrum sensing test statistic according to
    Pu Wang, Jun Fang, Ning Han, and Hongbin Li, Multiantenna-Assisted Spectrum Sensing for Cognitive Radio, Equation 13
    
    Parameters:
    :param x: received signal samples (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return test statistic (float32)
    """
    Rx = StatCovarianceMatrix(x, L)
    eigenvalues = numpy.linalg.eigvalsh(Rx)
    # The sum of the eigenvalues is the same as the trace of Rx
    return(eigenvalues[0] / sum(eigenvalues))

def CyclicAutocorrelationDetection(x):
    """
    CyclicAutocorrelationDetection calculates the time domain cyclostationarity of a set of samples according to
    Ning Han, Guanbo Zheng, Sung Hwan Sohn, and Jae Moung Kim, Cyclic Autocorrelation based Blind OFDM Detection and Identification for Cognitive Radio, Section III

    Parameters:
    :param x: signal samples (float32 list)
    
    Returns:
    :return ca: cyclic autocorrelation test statistic (float32)
    """
    A = numpy.var(x) # variance of the symbol sequence
    df = 10 # subcarrier spacing
    Tu = 1 / df # useful symbol duration
    Tg = 1E-6 # guard interval duration
    Ts = Tu + Tg # symbol duration Ts = Tu + Tg
    # todo
    pass
    
def SpectralCorrelationDetection(x):
    # todo
    pass

def BuildStepsList(ivs, vsi, fvs):
    """
    BuildStepsList builds a list containing numbers from a starting value up to a final value in given increments.
    This is useful for providing test cases for a simulation.
    
    Parameters:
    :param ivs: initial value step (numeric)
    :param vsi: value step increment (numeric)
    :param fvs: final value step (numeric)
    
    Returns:
    :return vss: value steps (numeric list)
    """
    # Initialize a list of steps as empty
    vss = []
    # Start at the initial number of steps given
    vs = ivs
    # Go through and add steps in the increments given to the steps list
    while(vs <= fvs):
        vss.append(vs)
        vs += vsi
    # Send back the steps list
    return(vss)

# Calculate Linear SNR from SNR in Decibels
# @jit
def LinearSNR(snr_db):
    """
    LinearSNR calculates the linear-scale (range (0, 1)) signal-to-noise ratio (SNR) from a given SNR given in decibels.
    
    Parameters:
    :param snr_db: signal-to-noise ratio in decibels (float_32)
    
    Return:
    :return: linear signal-to-noise ratio (float32, (0, 1))
    """
    return(10 ** (snr_db / 10))

def MCTrial(ssa, st, nct, snr_db, n, pfa, L):
    """ 
    MCTrial performs a Monte Carlo Trial, which is a True or False determination.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    :param n: number of samples (integer)
    :param L: covariance window size (integer)
    
    Returns:
    :return: True or False (boolean)
    """
    # Add the signal to the noise channel to produce our received signal x 
    x = GenerateSignal(st, snr_db, n) + GenerateNoise(nct, n)
        
    # Determine whether the test statistic exceeds the threshold
    gamma = GenerateTestStatistic(ssa, x, L)
    gamma_0 = GenerateThreshold(ssa, x, L, pfa)
    if(gamma >= gamma_0):
        outcome = True
    else:
        outcome = False
    return(outcome, gamma, gamma_0)
    
def GenerateTestStatistic(ssa, x, L):
    """
    GenerateTestStatistic generates the Test Statistic for a signal utilizing a given spectrum sensing method.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param x: received signal (float32 list)
    :param L: covariance window size (integer)
    
    Returns:
    :return gamma: test statistic (float32)
    """
    if(ssa == "ed"):
        gamma = EnergyDetection(x)
    elif(ssa == "cav"):
        gamma = CovarianceAbsoluteValueDetection(x, L)
    elif(ssa == "cfn"):
        gamma = CovarianceFrobeniusNormDetection(x, L)
    elif(ssa == "mme"):
        gamma = MaximumMinimumEigenvalueDetection(x, L)
    elif(ssa == "eme"):
        gamma = EnergyWithMinimumEigenvalueDetection(x, L)
    # If the method was not given or is invalid, just perform Energy Detection
    else:
        gamma = EnergyDetection(x)
    return(gamma)

def GenerateThreshold(ssa, x, L, pfa):
    """
    GenerateThreshold generates a Threshold for a signal utilizing a given spectrum sensing method.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param x: received signal (float32 list)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return gamma_0: threshold (float32)
    """
    if(ssa == "ed"):
        gamma_0 = EnergyThreshold(x, pfa)
    elif(ssa == "cav"):
        gamma_0 = CovarianceAbsoluteValueThreshold(x, L, pfa)
    elif(ssa == "cfn"):
        gamma_0 = CovarianceFrobeniusNormThreshold(x, L, pfa)
    elif(ssa == "mme"):
        gamma_0 = MaximumMinimumEigenvalueThreshold(x, L, pfa)
    elif(ssa == "eme"):
        gamma_0 = EnergyWithMinimumEigenvalueThreshold(x, L, pfa)
    # If the method was not given or is invalid, just perform Energy Detection
    else:
        gamma_0 = EnergyDetection(x)
    return(gamma_0)

def LoadLTESamples(snr_db):
    """
    LoadLTESamples loads LTE samples from a file into memory.
    
    Parameters:
    :param n: number of samples to load (integer)
    
    Returns:
    :return LTE samples (complex32)
    """
    file = open("C:/signalsamples/ltesample.dat", "r")
    x = []
    i = 0
    for line in file:
        i = i + 1
        line = line.replace("\n", "")
        line = line.replace("i", "j")
        line = line.replace(" ", "")
        value = LinearSNR(snr_db) * complex(line)
        x.append(value)
    return(x)

def LoadWIFISamples(snr_db):
    """
    LoadWIFISamples loads WIFI samples from a file into memory.
    
    Parameters:
    :param n: number of samples to load (integer)
    
    Returns:
    :return WIFI samples (complex32)
    """
    file = open("C:/signalsamples/wifi_80211ac_samples.dat", "r")
    x = []
    i = 0
    for line in file:
        #print(line)
        i = i + 1
        line = line.replace("\n", "")
        line = line.replace("i", "j")
        line = line.replace(" ", "")
        #print(line)
        value = LinearSNR(snr_db) * complex(line) * 1.0
        x.append(value)
    return(x)
    
def GenerateSignal(st, snr_db, n):
    """
    GenerateSignal generates a signal of a given type, signal-to-noise ratio, and number of samples.
    
    Parameters:
    :param st: signal type (string)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    
    Returns:
    :return: signal samples (float32 list)
    """
    # Convert the string to lowercase
    # Since the conditional statements that follow check for the lowercase string, this makes the function case-insensitive
    st = st.lower()
    # SNR is given in DB for ease of use, but the math uses linear SNR
    snr = LinearSNR(snr_db)
    # Check what kind of signal we want to use and generate it
    if(st == "gaussian"):
        loc = 0  # Mean or center of the distribution
        scale = 1  # Standard deviation of the distribution (a.k.a. sigma squared)
        return(numpy.sqrt(snr) * numpy.random.normal(loc, scale, n))
    # If signal type is invalid or not given just generate a Gaussian (white noise) type of signal
    else:
        return(numpy.sqrt(snr) * numpy.random.normal(0, 1, n))
    
# Generate Noise
# Arguments:
# nct: type of noise channel (string)
# n: number of samples to generate (integer)
# Output:
# noise: list of random numbers (float32)
def GenerateNoise(nct, n):
    """
    GenerateNoise generates a set of noise samples for a given type of noise channel and number of samples.
    
    Parameters:
    :param nt: noise channel type (string)
    :param n: number of samples to generate (integer)
    
    Returns:
    :return: noise samples (float32 list)
    """
    # Convert the string to lowercase
    # Since the conditional statements that follow check for the lowercase string, this makes the function case-insensitive
    nct = nct.lower()
    # Check what kind of noise channel we want to use and generate it
    if(nct == "gaussian"):
        loc = 0  # Mean or center of the distribution
        scale = 1  # Standard deviation of the distribution (a.k.a. sigma squared)
        return(numpy.random.normal(loc, scale, n))
    # If noise type is invalid or not given just generate a Gaussian (white noise) channel
    else:
        return(numpy.random.normal(0, 1, n))
    
def NOSTest(ssa, nct, inos, noss, fnos, nmc, L, pfa):
    """
    NOSTest performs a test of the performance of a given spectrum sensing algorithm over a varying number of samples.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param inos: initial number of samples (integer)
    :param noss: number of samples step (integer)
    :param fnos: final number of samples (integer)
    :param nmc: number of Monte Carlo trials per sample step (integer)
    :param L: covariance window size (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return Ns: number of samples steps (integer list)
    :return pds: probabilities of detection (float32 list)
    """
    Ns = BuildStepsList(inos, noss, fnos)
    pds = []
    # Iterate through each number of samples in the set
    for n in Ns:
        H = 0
        print("Number of samples: " + str(n))
        for mc in range(nmc):
            assert(mc >= 0)
            if(1 == 1):
                H = H + 1
        pds.append(H / nmc)
    return(Ns, pds)

def SNRTest(ssa, nct, isnr, snri, fsnr, nmc, L, n, pfa):
    """
    SNRTest performs a test of the performance of a given spectrum sensing algorithm over a varying signal-to-noise ratio (SNR).
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param isnr: initial signal-to-noise ratio (float32)
    :param snri: signal-to-noise ratio increment (float32)
    :param fsnr: final signal-to-noise ratio (float32)
    :param nmc: number of Monte Carlo trials per sample step (integer)
    :param L: covariance window size (integer)
    :param n: number of samples (integer)
    :param pfa: probability of false alarm (float32)
    
    Returns:
    :return snrs: signal-to-noise ratios (float32 list)
    :return pds: probabilities of detection (float32 list)
    """
    snrs = BuildStepsList(isnr, snri, fsnr)
    pds = []
    # Iterate through each SNR in the set
    for snr in snrs:
        H = 0
        print("SNR: " + str(snr))
        for mc in range(nmc):
            assert(mc >= 0)
            if(MCTrial("gaussian", "gaussian", "ed", snr, n, pfa, L)):
                H = H + 1
        pds.append(H / nmc)
    return(snrs, pds)

def RunTest(tt, ssa, st, nct, start, increment, end, nmc, L, n, snr_db, pfa):
    """
    RunTest performs the given test of the performance of a given spectrum sensing algorithm over a varying parameter (e.g. number of samples, SNR, etc.)
    
    Parameters:
    :param tt: test type (string)
        n: number of samples
        snr: signal-to-noise ratio
        L: covariance window size
        snr: signal-to-noise ratio
        roc, pfa: receiver operating characteristic
        (default): number of samples
    :param ssa: spectrum sensing algorithm (string)
        ed: Energy Detection
        cav: Covariance Absolute Value
        cfn: Covariance Frobenius Norm
        mme: Maximum-Minimum Eigenvalue
        eme: Energy with Minimum Eigenvalue
        agm: Arithmetic-to-Geometric Mean
        met: Maximum Eigenvalue to the Trace
        caf: Cyclic Autocorrelation Function
        scf: Spectral Correlation Function
        (default): Energy Detection
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param start: initial value of varying parameter (numeric)
    :param increment: increment of varying parameter per step (numeric)
    :param end: final value of varying parameter (numeric)
    :param nmc: number of Monte Carlo trials per step (integer)
    :param L: covariance window size (integer), ignored if using Energy Detection SSA
    :param n: number of samples(integer), ignored if performing number of samples test
    :param snr_db: signal-to-noise ratio in decibels (float32), ignored if performing signal-to-noise ratio test
    :param pfa: probability of false alarm (float32), ignored if performing receiver operating characteristic (ROC) test
    
    Returns:
    :return pds: probabilities of detection (float32 list)
    :return steps: steps of the varying parameter (numeric list)
    """
    steps = BuildStepsList(start, increment, end)
    pds = []
    assert(len(pds) == 0)
    test_type = ""
    if(tt == "n"):
        test_type = "Number of Samples"
    elif(tt == "snr"):
        test_type = "Signal-to-Noise Ratio"
    elif(tt == "roc"):
        test_type = "Receiver Operating Characteristic"
    elif(tt == "pfa"):
        test_type = "Probability of False Alarm (Threshold Function)"
    else:
        test_type = "Number of Samples"
    print("Running test: " + test_type)
    pds = []
    gammas = []
    gammas_0 = []
    for step in steps:
        H_1s = 0
        gamma_sum = 0
        gamma_0_sum = 0
        print("Performing " + str(nmc) + " Monte Carlo trials for step " + str(step))
        for nmc in range(nmc):
            nmc = nmc
            gamma = 0
            gamma_0 = 0
            outcome = 0
            outcome, gamma, gamma_0 = MCTrial(ssa, st, nct, snr_db, n, step, L)
            gamma_sum = gamma_sum + gamma
            gamma_0_sum = gamma_0_sum + gamma_0
            if(outcome == True):
                H_1s = H_1s + 1
        pds.append(H_1s / nmc)
        gammas.append(gamma_sum / nmc)
        gammas_0.append(gamma_0_sum / nmc)
    return(pds, steps, gammas, gammas_0)

def GenerateGamma(ssa, x, L):
    """
    GenerateGamma generates the test statistic for the given signal using the specified spectrum sensing algorithm.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param x: received signal (float32 list)
    :param L: covariance window size, only used in covariance-based methods, ignored otherwise (int32)
    
    Returns:
    :return gamma, also known as the test statistic (float32)
    """
    ssa = ssa.lower()
    if(ssa == "ed"):
        return(EnergyDetection(x))
    elif(ssa == "cav"):
        return(CovarianceAbsoluteValueDetection(x, L))
    elif(ssa == "cfn"):
        return(CovarianceFrobeniusNormDetection(x, L))
    elif(ssa == "mme"):
        return(MaximumMinimumEigenvalueDetection(x, L))
    elif(ssa == "eme"):
        return(EnergyWithMinimumEigenvalueDetection(x, L))
    else:
        return(EnergyDetection(x))
    
def GenerateRatings(ssa, st, nct, L, nt, snr_db, n):
    """
    GenerateRatings generates a list of test statistics and a list of binary signal present/not present values.
    
    Parameters:
    :param ssa: specturm sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param L: covariance window size (int32)
    :param nt: number of Monte Carlo trials (int32)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    :param n: number of samples per trial (int32)
    
    Returns:
    :return tss: test statistics (float32 list)
    :return sps: signal presences (int32 list)
    """
    st = st.lower()
    nct = nct.lower()
    tss = []
    sps = []
    samples = []
    # case signal not present
    if(st == "lte"):
        samples = LoadLTESamples(snr_db)
    elif(st == "wlan" or st == "wifi"):
        samples = LoadWIFISamples(snr_db)
        #todo add wifi function
    for i in range(0, nt):
        #print(i)
        x = GenerateNoise(nct, n)
        gamma_np = GenerateGamma(ssa, x, L)
        tss.append(gamma_np)
        sps.append(0)
        if(st == "lte"):
            floor = numpy.random.randint(0, len(samples) - n)
            ceiling = floor + n
            y = samples[floor : ceiling] + GenerateNoise(nct, n)
        else:
            y = GenerateSignal(st, snr_db, n) + GenerateNoise(nct, n)
        gamma_p = GenerateGamma(ssa, y, L)
        tss.append(gamma_p)
        sps.append(1)
    return(tss, sps)

def CalculateFPG(ssa, n, st, nct, nmc, snr_db, pfa, L):
    """
    CalculateFPG calculates the false positive gamma for a given spectrum sensing algorithm.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param n: number of samples (int32)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param nmc: number of Monte Carlo trials (int32)
    
    Returns:
    :return average gamma (float32 list)
    """
    accumulated_gamma = 0
    for mc in range(nmc):
        assert(mc >= 0)
        x = GenerateNoise(nct, n)
        gamma = GenerateGamma(ssa, x, L)
        accumulated_gamma = accumulated_gamma + gamma
    return(accumulated_gamma / nmc)

def CalculateTPG(ssa, n, st, nct, nmc, snr_db, pfa, L):
    """
    CalculateTPG calculates the true prositive gamma for a given spectrum sensing algorithm.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param n: number of samples (int32)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param nmc: number of Monte Carlo trials (int32)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    
    Returns:
    :return average gamma (float32 list)
    """
    accumulated_gamma = 0
    for mc in range(nmc):
        assert(mc >= 0)
        x = GenerateSignal(st, snr_db, n) + GenerateNoise(nct, n)
        gamma = GenerateGamma(ssa, x, L)
        accumulated_gamma = accumulated_gamma + gamma
    return(accumulated_gamma / nmc)

def CalculateFPR(ssa, n, nct, nmc, c, L):
    """
    CalculateFPR calculates the false positive rate for a given spectrum sensing algorithm.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param n: number of samples (int32)
    :param nct: noise channel type (string)
    :param nmc: number of Monte Carlo trials (int32)
    :param c: threshold (float32)
    :param L: covariance window size (int32)
    
    Returns:
    :return fpr: false positive rate (float32)
    """
    fpr = 0
    for mc in range(nmc):
        assert(mc >= 0)
        x = GenerateNoise(nct, n)
        gamma = GenerateGamma(ssa, x, L)
        if(gamma >= c):
            fpr = fpr + 1
    return(fpr / nmc)

def CalculateTPR(ssa, n, st, nct, nmc, snr_db, c, L):
    """
    Calculate TPR calculates the true positive rate for a given spectrum sensing algorithm.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param n: number of samples (int32)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param nmc: number of Monte Carlo trials (int32)
    :param snr_db: signal-to-noise ratio (float32)
    :param c: threshold (float32)
    :param L: covariance window size (int32)
    
    Returns:
    :return tpr: true positive rate (float32)
    """
    tpr = 0
    for mc in range(nmc):
        assert(mc >= 0)
        ps = GenerateSignal(st, snr_db, n)
        fs = GenerateNoise(nct, n)
        x = ps + fs 
        gamma = GenerateGamma(ssa, x, L)
        if(gamma >= c):
            tpr = tpr + 1
    return(tpr / nmc)

def CalculateRatings(ssa, st, nct, L, nmc, c, n, snr_db):
    """
    CalculateRatings provides class 1 (false positive ratio) and class 2 (true positive ratio) rating for a Monte Carlo trial of an ROC test.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param nmc: number of Monte Carlo trials (int32)
    :param c: threshold (float32)
    :param L: covariance window size (int32)
    :param n: number of samples (int32)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    
    Returns:
    :return fpr: false positive ratio (float32)
    :return tpr: true positive ratio (float32)
    """
    fpr = CalculateFPR(ssa, n, nct, nmc, c, L)
    tpr = CalculateTPR(ssa, n, st, nct, nmc, snr_db, c, L)
    return(fpr, tpr)
    
def ROCTest(tt, ssa, st, nct, start, increment, end, nmc, L, n, snr_db):
    """
    ROCTest calculates the probabilities of detection and probabilities of false alarm for a
    given spectrum sensing algorithm (SSA) with given related received signal parameters.
    
    Parameters:
    :param tt: test type (string)
        n: number of samples
        snr: signal-to-noise ratio
        L: covariance window size
        snr: signal-to-noise ratio
        roc, pfa: receiver operating characteristic
        (default): number of samples
    :param ssa: spectrum sensing algorithm (string)
        ed: Energy Detection
        cav: Covariance Absolute Value
        cfn: Covariance Frobenius Norm
        mme: Maximum-Minimum Eigenvalue
        eme: Energy with Minimum Eigenvalue
        agm: Arithmetic-to-Geometric Mean
        met: Maximum Eigenvalue to the Trace
        caf: Cyclic Autocorrelation Function
        scf: Spectral Correlation Function
        (default): Energy Detection
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param start: initial value of varying parameter (numeric)
    :param increment: increment of varying parameter per step (numeric)
    :param end: final value of varying parameter (numeric)
    :param nmc: number of Monte Carlo trials per step (integer)
    :param L: covariance window size (integer), ignored if using Energy Detection SSA
    :param n: number of samples(integer), ignored if performing number of samples test
    :param snr_db: signal-to-noise ratio in decibels (float32), ignored if performing signal-to-noise ratio test
    :param pfa: probability of false alarm (float32), ignored if performing receiver operating characteristic (ROC) test
    
    Returns:
    :return X: probabilities of detection (float32 list)
    :return Y: probabilities of false alarm (float32 list)
    :return steps: steps of the varying parameter (numeric list)
    """
    steps = BuildStepsList(start, increment, end)
    #print("steps: " + str(steps))
    Xs = []
    Ys = []
    for c in steps:
        #print(str(c))
        X, Y = CalculateRatings(ssa, st, nct, L, nmc, c, n, snr_db)
        Xs.append(X)
        Ys.append(Y)
    return(Xs, Ys)

def AUCTest(ssa, st, nct, snr_db, n, pfa, L):
    """
    AUCTest performs several ROC tests with changing conditions
    """

def MCTrialSignalPresent(ssa, st, nct, snr_db, n, pfa, L):
    """ 
    MCTrial performs a Monte Carlo Trial, which is a True or False determination.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    :param n: number of samples (integer)
    :param L: covariance window size (integer)
    
    Returns:
    :return: True or False (boolean)
    """
    # Add the signal to the noise channel to produce our received signal x 
    x = GenerateSignal(st, snr_db, n) + GenerateNoise(nct, n)
        
    # Determine whether the test statistic exceeds the threshold
    gamma = GenerateTestStatistic(ssa, x, L)
    gamma_0 = GenerateThreshold(ssa, x, L, pfa)
    if(gamma >= gamma_0):
        outcome = True
    else:
        outcome = False
    return(outcome, gamma, gamma_0)

def MCTrialSignalAbsent(ssa, nct, snr_db, n, pfa, L):
    """ 
    MCTrial performs a Monte Carlo Trial, which is a True or False determination.
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
    :param nct: noise channel type (string)
    :param snr_db: signal-to-noise ratio in decibels (float32)
    :param n: number of samples (integer)
    :param L: covariance window size (integer)
    
    Returns:
    :return: True or False (boolean)
    """
    # Generate just noise since there is no signal
    x = GenerateNoise(nct, n)
        
    # Determine whether the test statistic exceeds the threshold
    gamma = GenerateTestStatistic(ssa, x, L)
    gamma_0 = GenerateThreshold(ssa, x, L, pfa)
    if(gamma >= gamma_0):
        outcome = True
    else:
        outcome = False
    return(outcome, gamma, gamma_0)

def FastDeLong(X, Y):
    """
    FastDeLong calculates the area under a receiver operating characteristic (ROC) curve (AUC)
    Python implementation by arr1
    
    % fastDeLong.m
    % For ROC analysis, implements a fast rank-based algorithm for the 
    % Mann-Whitney AUC estimator and for the DeLong covariance matrix estimator. 
    % References:
    %
    % E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the
    % areas under two or more correlated receiver operating characteristic
    % curves: A nonparametric approach," Biometrics, vol. 44, no. 3,
    % pp. 837-845, Sept. 1988.  
    % 
    % X. Sun and W. Xu, "Fast implementation of DeLong's algorithm for
    % comparing the areas under correlated receiver operating characteristic
    % curves", IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389-1393,
    % Nov. 2014.
    %
    % Note: The meanings of x and y are flipped relative to the above references.
    %
    % Inputs:
    %   X (q x m matrix of class 1 ratings)
    %   Y (q x n matrix of class 2 ratings)    
    % 
    % Outputs: 
    %   AUC (q x 1 vector of AUC estimates)
    %   S (q x q covariance matrix) 
    %
    % Adam Wunderlich 9/29/2014
    
    q is total classifiers
    
    Parameters:
    :param X: (1 × m matrix of class 1 ratings) scores for m signal absent cases 
    :param Y: (1 × n matrix of class 2 ratings) scores for n signal present cases
    
    Returns:
    :return AUC: (q × 1 vector of AUC estimates)
    :return S: (q × q covariance matrix)
    """
    q, m = (1, len(X))
    q, n = (1, len(Y))
    print(q)
    print(m)
    print(n)
    V10 = []
    V01 = []
    xi = range((n), (m + n))
    print(xi)
    yj = range(0, n)
    print(yj)
    for k in range(0, q):
        x = X[k:]
        y = Y[k:]
        z = x + y
        TX = scipy.stats.rankdata(x)
        print(TX)
        TY = scipy.stats.rankdata(y)
        print(TY)
        TZ = scipy.stats.rankdata(z)
        print(TZ)
        V10[:k] = 1 - (TZ[xi] - TX) / n
        print(V10)
        V01[:k] = (TZ[yj] - TY) / m
    AUC = numpy.mean(V01)
    S = (numpy.cov(V10) / m) + (numpy.cov(V01) / n)
    return(AUC, S)

def npAUC_CI(alpha1, alpha2, X, Y):
    """
    npAUC_CI is a Python adaptation of npAUC_CI.m, a MATLAB script
    
    Parameters:
    :param alpha1: lower significance level (float32)
    :param alpha2: higher significance level (float32)
    :param X: q×m matrix of class 1 [signal absent] ratings (float32 matrix)
    :param Y: q×n matrix of class 2 [signal present] ratings (float 32 matrix)
    
    Returns:
    :return AUC: area under the curve point estimates (float32 list)
    :return AUC_CI: confidence interval for the area under a curve when q == 1 or CI for AUC difference when q == 2 (float32)
    
    % npAUC_CI.m
    % For an ROC assessment, returns a 1-(alpha1+alpha2) confidence interval 
    % for a single AUC or for a difference of AUCs.  This function requires the 
    % function fastDeLong.m, and assumes that variabilty is due to cases only.  
    %
    % The confidence interval for a single AUC is computed using the logit
    % transformation method recommended on page 107 in the book: 
    % Pepe M.S.,"The statistical evaluation of medical tests for classification 
    % and prediction", Oxford Univ. Press, 2003.  For a difference of AUCs, 
    % the usual normal approximation is used.  
    %
    % Inputs:  alpha1 (lower significance level), alpha2 (upper significance level) 
    % (for conventional two-sided 95% intervals, set alpha1=alpha2=0.025), 
    % X (q x m matrix of class 1 ratings)
    % Y (q x n matrix of class 2 ratings)
    % where q = 1 or 2, the number of fixed imaging scenarios or readers
    %
    % Outputs: AUC (AUC point estimates), AUC_CI (confidence interval for 
    % a single AUC (q=1) or for a difference (q=2))
    % Note: The interval estimate for a difference is for
    % performance of the second scenario minus the first.  
    % 
    % Adam Wunderlich
    % 9/26/2014
    """
    q, n = (len(Y), len(Y[0]))
    assert(n == n)
    AUC, S = FastDeLong(X, Y)
    AUC_CI = []
    if(q == 1):
        # use logit transformation method
        logit_CI = []
        if((alpha1 != 0) and (alpha2 != 0)):
            logit_CI[1] = numpy.log(AUC / (1 - AUC)) + scipy.stats.norm.ppf(alpha1) * numpy.sqrt(S) / (AUC * (1 - AUC))
            logit_CI[2] = numpy.log(AUC / (1 - AUC)) + scipy.stats.norm.ppf(1 - alpha2) * numpy.sqrt(S) / (AUC * (1 - AUC))
            AUC_CI = numpy.exp(logit_CI) / (1 + numpy.exp(logit_CI))
        elif((alpha1 == 0) and (alpha2 != 0)):
            logit_CI[2] = numpy.log(AUC / 1 - AUC) + scipy.stats.norm.ppf(1 - alpha2) * numpy.sqrt(S) / (AUC * (1 - AUC))
            AUC_CI[1] = 0
            AUC_CI[2] = numpy.exp(logit_CI[2]) / (1 + numpy.exp(logit_CI(2)))
        elif((alpha1 != 0) and (alpha2 == 0)):
            logit_CI[1] = numpy.log(AUC / (1 - AUC)) + scipy.stats.norm.ppf(alpha1) * numpy.sqrt(S) / (AUC * (1 - AUC))
            AUC_CI[1] = numpy.exp(logit_CI[1]) / (1 + numpy.exp(logit_CI[1]))
            AUC_CI[2] = 1
    elif(q == 2):
        # use usual normal approximation
        AUCdiff = AUC[2] - AUC[1]
        var_diff = S[1][1] + S[2][2] - (2 * S[1][2])
        if(alpha1 == 0):
            AUC_CI[1] = -1
        else:
            AUC_CI[1] = AUCdiff + scipy.stats.norm.ppf(alpha1) * numpy.sqrt(var_diff)
        if(alpha2 == 0):
            AUC_CI[2] = 1
        else:
            AUC_CI[2] = AUCdiff + scipy.stats.norm.ppf(1 - alpha2) * numpy.sqrt(var_diff)
            
def SNRTSTest(ssa, st, nct, start, increment, end, nmc, L, n):
    """
    
    Parameters:
    :param ssa: spectrum sensing algorithm (string)
        ed: Energy Detection
        cav: Covariance Absolute Value
        cfn: Covariance Frobenius Norm
        mme: Maximum-Minimum Eigenvalue
        eme: Energy with Minimum Eigenvalue
        agm: Arithmetic-to-Geometric Mean
        met: Maximum Eigenvalue to the Trace
        caf: Cyclic Autocorrelation Function
        scf: Spectral Correlation Function
        (default): Energy Detection
    :param st: signal type (string)
    :param nct: noise channel type (string)
    :param start: initial SNR value in dB (float32)
    :param increment: incremental SNR value in dB (float32)
    :param end: final SNR value in dB (float32)
    :param nmc: number of Monte Carlo trials per step (int)
    :param L: covariance window size (int)
    :param n: number of samples (int)
    
    Returns:
    :return tss: test statistic values (float32 list)
    """
    snrs = BuildStepsList(ivs = start, vsi = increment, fvs = end)
    tss = []
    ssa = ssa.lower()
    for snr in snrs:
        ts = 0
        x = GenerateSignal(st, snr, n) + GenerateNoise(nct, n)
        if(ssa == "ed"):
            ts = EnergyDetection(x)
        elif(ssa == "cav"):
            ts = CovarianceAbsoluteValueDetection(x, L)
        elif(ssa == "cfn"):
            ts = CovarianceFrobeniusNormDetection(x, L)
        elif(ssa == "mme"):
            ts = MaximumMinimumEigenvalueDetection(x, L)
        elif(ssa == "eme"):
            ts = EnergyWithMinimumEigenvalueDetection(x, L)
        elif(ssa == "agm"):
            ts = ArithmeticToGeometricMeanDetection(x, L)
        elif(ssa == "met"):
            ts = MaximumEigenvalueTraceDetection(x, L)
        elif(ssa == "caf"):
            pass

def calculate_roc (data, labels):
    """
    calculate_roc calculates true positive rates and false positive rates to build an ROC curve.
    It also calculates the area under the curve (AUC) and the AUC-based symmetry of the ROC curve.
    Thanks to Max Lees (william.lees@nist.gov) for providing the fast ROC and AUC functions.
    Modified by André Rosete (andre.rosete@nist.gov) to calculate symmetry.
        
    Parameters:
    :param data: true positive rates (float32 list)
    :param labels: signal present or not present (int32 list)
        
    Returns:
    None. Plots ROC curve and line.
    """
    #print(data)
    #print(labels)
    labels = numpy.array(labels)
    total_samples = len(labels)
    total_positive = numpy.sum(labels)
    if total_positive == 0:
        return
    total_negative = total_samples - total_positive
    indices = numpy.argsort(data)
    labels = labels[numpy.flipud(indices)]
    true_positive_rate = []
    false_positive_rate = []
    current_true_positives = total_positive
    current_false_positives = total_negative
    true_positive_rate.append(1.0)
    false_positive_rate.append(1.0)
    for index in range(total_samples):
        if labels[index] == 1:
            current_true_positives = current_true_positives - 1
        else:
            current_false_positives = current_false_positives - 1
        true_positive_rate.append(current_true_positives / total_positive)
        false_positive_rate.append(current_false_positives / total_negative)
    linear = numpy.linspace(0, 1)
    g = []
    g.append(0.0)
    auc = 0.0
    auc1 = 0.0
    auc2 = 0.0
    for i in range(0, len(false_positive_rate) - 1):
        g.append(1 - false_positive_rate[i])
        height = true_positive_rate[i]
        width = false_positive_rate[i] - false_positive_rate[i + 1]
        area = height * width
        auc = auc + area
        if(height >= g[i]):
            height2 = height - g[i]
            area2 = height2 * width
            auc2 = auc2 + area2
    auc_or = auc
    if(auc_or == 0):
        auc_or = 1e-6
        auc1 = 0.5
        auc2 = 0.5
    else:
        auc1 = auc - auc2
        auc = 1 - auc
        auc1 = (auc1 / auc_or) * auc
        auc2 = (auc2 / auc_or) * auc
    
    symmetry = 1 - abs((auc1 - auc2) / 2)
    
#     for i in range(len(true_positive_rate)-1):
#         auc = auc + ((true_positive_rate[i]-true_positive_rate[i+1])*false_positive_rate[i])
#     print('AUC: %s' % (auc))

#     plt.plot(true_positive_rate, false_positive_rate)
#     plt.plot(g, false_positive_rate)
#     plt.plot(linear, linear)
#     plt.title("SSA Receiver Operating Characteristic")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.show()
    return(auc, symmetry, true_positive_rate, false_positive_rate)
    
def auc_vs_n(ssa, st, nct, L, nt, auct, snr_db, N):
    auc = []
    sym = []
    average_auc = 0.0
    average_sym = 0.0
    for n in N:
        print("Number of samples: " + str(n))
        data, labels = GenerateRatings(ssa, st, nct, L, nt, snr_db, n)
        current_auc, current_sym, tpr, fpr = calculate_roc(data, labels)
        average_auc = average_auc + current_auc
        average_symmetry = average_sym + current_sym
        auc.append(current_auc)
        sym.append(current_sym)
    #plt.plot(N[1:], auc[1:])
    #plt.plot(N[1:], sym[1:])
    #plt.axhline(y = 0.5, color = 'r', linestyle = '-')
    #plt.title("AUC vs. N")
    #plt.xlabel("Number of Samples")
    #plt.ylabel("Area under the Curve")
    #plt.show()
    return(auc, sym)
    
def SNR_auc_vs_n(ssa, st, nct, L, nt, auct, snr_dbs, N):
    snr_dbs = BuildStepsList(snr_dbs[0], snr_dbs[1], snr_dbs[2])
    N = BuildStepsList(N[0], N[1], N[2])
    print(snr_dbs)
    for snr_db in snr_dbs:
        print("SNR (dB): " + str(snr_db))
        
        auc, sym = auc_vs_n(ssa, st, nct, L, nt, auct, snr_db, N)
        plt.plot(N[1:], auc[1:], label = "SNR (dB): " + str(snr_db))
    
    plt.title(ssa + ", " + st + ", " + nct + ", L = " + str(L) + ", MCT = " + str(nt) + ", AUC vs. N")
    plt.xlabel("Number of Samples")
    plt.ylabel("Area Under ROC Curve")
    plt.legend()
    plt.show()

SNR_auc_vs_n(ssa = "cav", st = "wifi", nct = "Gaussian", L = 10, nt = 10000, auct = 1, snr_dbs = [-12, 3, 12], N = [100, 1000, 10000])
# data, labels = GenerateRatings(ssa = "ed", st = "lte", nct = "Gaussian", L = 100, nt = 1000, snr_db = 10, n = 1000)
# calculate_roc(data, labels)
