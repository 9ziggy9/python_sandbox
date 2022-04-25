def average(*nums):
    try:
      return sum(nums) / len(nums)
    except:
      return -1

def sample_variance(*nums):
    avg = average(*nums)
    sV = sum([(num - avg) ** 2 for num in nums]) / (len(nums) - 1)
    return sV
