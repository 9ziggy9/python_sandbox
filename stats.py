import math
import collections

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

def v_median(*nums):
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

def z_score(x, mu, sigma):
    return (x - mu) / sigma
