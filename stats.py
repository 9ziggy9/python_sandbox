def average(*nums):
    try:
      return sum(nums) / len(nums)
    except:
      return -1

def sample_variance(*nums):
    avg = average(*nums)
    sV = sum([(num - avg) ** 2 for num in nums]) / (len(nums) - 1)
    return sV

# returns value that must be appended to list to force a given avg
def force_average_with(target, *nums):
    avg = average(*nums)
    # (avg * len(nums) + new_number) / (len(nums) + 1) = target
    return (len(nums) + 1) * target - avg * len(nums)
