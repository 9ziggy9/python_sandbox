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

def generate_s_space(*states, subspace=[]):
    if len(states) == 1: states[0]
    a,b = states[0]
    return [
    ]

def product_space(states1, states2):
    space = []
    for x in states1:
        for y in states2:
            if x and y:
                space.append(x+y)
    return space

# Default arguments are evaluated when the function definition is encountered,
# that is, when Python sees the def line. They are then BOUND to the function.
# With mutable default arguments, the default arguments bound to the function
# may bechanged, which can lead to some surprising results. The use of None as
# default which is tested with conditional is a common pattern to overcome this
# behavior. Seems that this is perhaps analogous to a singleton or closure.
def explicit_stack(stacks, subspace=None):
    if subspace is None: subspace = []
    if len(stacks) == 0: return subspace
    print(stacks)
    subspace.append(stacks[0])
    return explicit_stack(stacks[1:], subspace)
