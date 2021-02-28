from math import sin, cos, pi
import numpy as np
import time
from numba import cuda
from numba import numba
from matplotlib import pyplot as plt

@numba.jit
def DFT_sequential(samples, frequencies):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies.shape[0]):
        for n in range(N):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / N) - sin(2 * pi * k * n / N) * 1j)

def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments)
    :param number: Number of function calls in a single averaging interval
    :param repeat: Number of repetitions
    :return: List of timing results or a single value if repeat is equal to one
    """

    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel()
            cuda.synchronize() # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)
    return times[0] if len(times)==1 else times

@cuda.jit
def DFT_kernel_frequency(samples, frequencies, k):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]
    frequencies[k] += sample * (cos(2 * pi * k * x / N) - sin(2 * pi * k * x / N) * 1j)

#Opgave1
@cuda.jit
def DFT_kernel_parallel(samples, frequencies):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    for n in range(N):
        frequencies[x] += samples[n] * (cos(2 * pi * x * n / N) - sin(2 * pi * x * n / N) * 1j)

@cuda.jit
def DFT_kernel_serie(samples, frequencies):
    for k in range(frequencies.shape[0]):
        for n in range(N):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / N) - sin(2 * pi * k * n / N) * 1j)

@cuda.jit
def DFT_kernel_between(samples, frequencies, threads):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    for i in range(x,x+len(frequencies),threads):
        for n in range(N):
            frequencies[i] += samples[n] * (cos(2 * pi * i * n / N) - sin(2 * pi * i * n / N) * 1j)




# Define the sampling rate and observation time
SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S


# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05


# Initiate the empty frequency components
frequencies = np.zeros(int(N/2+1), dtype=np.complex)


# Time the sequential CPU function
t = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10 )
print("CPU  ", t)

frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_parallel[5,51](sig_sum, frequencies1)
t2 = synchronous_kernel_timeit( lambda: DFT_kernel_parallel[5,51](sig_sum, frequencies1), number=10)
print("Kernel parallel  ",t2)

frequencies2 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_between[2,5](sig_sum, frequencies2,10)
t3 = synchronous_kernel_timeit( lambda: DFT_kernel_between[2,5](sig_sum, frequencies2,10), number=10)

print("Kernel in between threads  " , t3)

'''
frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
start = time.time()
DFT_kernel_parallel[5,51](sig_sum, frequencies1)
print(time.time() - start)
frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
start = time.time()
DFT_kernel_parallel[5,51](sig_sum, frequencies1)
print(time.time() - start)
'''

frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_serie[1,1](sig_sum, frequencies1)
t2 = synchronous_kernel_timeit( lambda: DFT_kernel_serie[1,1](sig_sum, frequencies1), number=10)
print("Kernel 1 thread  " , t2)

frequencies3 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_serie[1,1] (sig_sum, frequencies3)

# Reset the results and run the DFT
frequencies = np.zeros(int(N/2+1), dtype=np.complex)
DFT_sequential(sig_sum, frequencies)

frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_parallel[5,51](sig_sum, frequencies1)

frequencies2 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_kernel_between[2,5](sig_sum, frequencies2,10)

# Plot to evaluate whether the results are as expected
fig, (ax1, ax2, ax5, ax3,ax4) = plt.subplots(1, 5)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

# Plot the frequency components
ax2.plot( xf, abs(frequencies), color='C3')
ax2.set_title('Sequential (CPU)')
ax5.plot(xf, abs(frequencies3),color='C3')
ax5.set_title('Serial')
ax3.plot( xf, abs(frequencies2), color='C3')
ax3.set_title('In between')
ax4.plot(xf, abs(frequencies1), color='C3')
ax4.set_title('Parallel')


plt.show()


