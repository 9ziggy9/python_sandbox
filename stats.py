import math
import collections

def average(*nums):
    try:
      return sum(nums) / len(nums)
    except:
      return -1

def sample_variance(*nums):
    avg = average(*nums)
    sV = sum([(num - avg) ** 2 for num in nums]) / len(nums)
    return sV

def standard_deviation(*nums):
    return math.sqrt(sample_variance(*nums))

# returns value that must be appended to list to force a given avg
def v_force_average(target, *nums):
    avg = average(*nums)
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
