import numpy as np
import math
import collections
from functools import reduce

def mean(*nums):
    try:
      return sum(nums) / len(nums)
    except:
      return -1

def sum_squares(*nums):
    avg = mean(*nums)
    return sum([(num - avg) ** 2 for num in nums])

def sample_variance(*nums):
    ss = sum_squares(*nums)
    return ss / (len(nums) - 1)

def population_variance(*nums):
    ss = sum_squares(*nums)
    return ss / len(nums)

def population_standard_deviation(*nums):
    return math.sqrt(population_variance(*nums))

def sample_standard_deviation(*nums):
    return math.sqrt(sample_variance(*nums))

# returns value that must be appended to list to force a given avg
def v_force_average(target, *nums):
    avg = mean(*nums)
    # (avg * len(nums) + new_number) / (len(nums) + 1) = target
    return (len(nums) + 1) * target - avg * len(nums)

def median(*nums):
    sorted_nums = sorted(nums)
    center = len(nums) / 2
    if (len(nums) % 2) == 0:
        left = int(center - 1)
        right = int(center)
        return (sorted_nums[left] + sorted_nums[right]) / 2
    else:
        return sorted_nums[int(center)]

def vs_modes(*nums):
    pms = [(n,count) for n, count in collections.Counter(nums).items() if count > 1]
    max_count = max(pms, key=lambda item:item[1])[1]
    modes = [m for m, count in pms if count == max_count]
    return modes

# application of Chebyshev's Theorem, at least 75% of data values lie within 2 sds
# of the mean
def cheby_range(mu,sigma):
    return (mu - 2 * sigma, mu + 2 * sigma)

# empirical rule, aka three-sigma rule; 1. 68% of observations fall within first sd,
# 95% within the first two sds, 99.7% within the first three.
def three_sigma_range(mu, sigma):
    return {
        '68%': (mu - sigma, mu + sigma),
        '95%': (mu - 2*sigma, mu + 2*sigma),
        '99.7%': (mu - 3*sigma, mu + 3*sigma)
    }

def find_nearest(*nums, val):
    snums = sorted(nums)
    nparr = np.asarray(snums)
    i = (np.abs(nparr - val)).argmin()
    return nparr[i]

def quartiles(*nums):
    snums = sorted(nums)
    q_2 = math.ceil(len(snums) / 2)
    q_1 = math.ceil(q_2 / 2)
    q_3 = math.ceil(q_2 + q_1)
    return { # have to subtract one to get arr index
            'q1': snums[q_1 - 1],
            'q2': snums[q_2 - 1],
            'q3': snums[q_3 - 1]
    }

def z_score(x, mu, sigma): # aka number of sds away from mean
    return (x - mu) / sigma

def percentile(n,N): #ordinal number n, total number N
    return 100 * (n/N)

def percentile_from_data(p, *nums):
    snums = sorted(nums)
    n = len(snums)
    l = math.ceil((p / 100) * n)
    return snums[l-1]

def ordinal_from_percentile(p,N):
    return math.ceil(p * N / 100)

def cv_from_data(*nums):
    sigma = sample_standard_deviation(*nums)
    mu = mean(*nums)
    return (sigma / mu) * 100

def mean_from_f(*points):
    fm2_table = []
    n = sum([p[2] for p in points])
    for point in points:
        fm2_table.append(point[2] * ((point[1] + point[0])/2))
    return sum(fm2_table) / n

def sample_variance_from_f(*points):
    fm2_table = []
    mu = mean_from_f(*points)
    n = sum([p[2] for p in points])
    for point in points:
        fm2_table.append(point[2] * ((point[1] + point[0])/2)**2)
    return (sum(fm2_table) - n*(mu**2))/ (n-1)

# PROBLEM: For a set of random variables, we associate to each a set of
# mutually exclusive events represented by characters. The number of possible
# events is arbitrary. Given a collection of such sets, generate the sample
# space of all possible measurements on the set of random variables.

# For a random variable X, suoose we have ('a','b','c')
# For Y, we could have ('d','e')
# A call to our desired procedure, say,
# generate_sample_space([('a','b','c'),('d',e')])
# should return the set [('ad', 'ae', 'bd', 'be', 'cd', 'ce')]
# Note that in general, the number of random variables provided is arbitary.

# The simple Cartesian product on tuples of characters, will be applied
# recursively to a growing subspace in our generate_sample_space() procedure.
def product_space(states1, states2):
    if len(states1) == 0: return list(states2)
    if len(states2) == 0: return list(states1)
    return [x+y for x in states1 for y in states2]

# Default arguments are evaluated when the function definition is encountered,
# that is, when Python sees the def line. They are then BOUND to the function.
# With mutable default arguments, the default arguments bound to the function
# may be changed, which can lead to some surprising results. The use of None as
# default which is tested with conditional is a common pattern to overcome this
# behavior.
def generate_sample_space(states, subspace=None):
    if subspace is None: subspace = []
    if len(states) == 0: return subspace
    subspace = product_space(subspace,states[0])
    return generate_sample_space(states[1:], subspace)

def compute_frequency(interval,data):
    (a,b) = interval
    return len([d for d in data if (d <= b and d >= a)])

def compute_cumulative_fs(w,interval,data):
    (a,b) = interval
    fs = []
    for x in range(a,b-1,w+1):
        fs.append(compute_frequency((x,x+w), data))
    return fs

def compute_ev(*data):
    return sum([x*P for (x,P) in data])

def variance_from_pd(*data):
    mu = compute_ev(*data)
    return sum([P*(x-mu)**2 for (x,P) in data])

def sd_from_pd(*data):
    return math.sqrt(variance_from_pd(*data))

def binomial_mass(n,x,p):
    return math.comb(n,x)*((p**x)*((1-p)**(n-x)))

def poisson_mass(k,r):
    return ((r**k)*math.exp(-r))/math.factorial(k)

# See wiki for more details
def hypergeometric_mass(N,K,n,k):
    return (math.comb(K,k)*math.comb(N-K, n-k))/math.comb(N,n)

# Probability P(x) that random variable X will take on value less than or
# equal to x.
def cdf_normal(x,mu,sigma):
    return (0.5)*(1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

# Compute probability that porportion of samples will fall under a given
# percentage.
def cdf_normal_from_pn(p_target,p,n):
    sigma_p = math.sqrt((p*(1-p))/n)
    z_score = (p_target - p) / sigma_p
    return (0.5)*(1 + math.erf(z_score/math.sqrt(2)))

def confidence_interval(x,sigma,n,z):
    return (x-z*(sigma/math.sqrt(n)),x+z*(sigma/math.sqrt(n)))
