# datetime:2021/3/8 15:52
# software: PyCharm

"""
File description：

"""


# Leetcode 入门级题目
class Solution(object):
	# 1.刪除重复元素（倒序刪除尾部元素）
	def removeDuplicates(self, nums):
		"""
		:type nums: List[int]
		:rtype: int
		"""
		for i in range(len(nums) - 1, 0, -1):
			if nums[i] == nums[i - 1]:
				del nums[i]

		print(nums)
		return len(nums)

	# 2.股票买卖问题
	def maxProfit(self, prices):
		"""
		:type prices: List[int]
		:rtype: int
		"""
		i = 0
		sum = 0
		while True:
			# 当index指向末尾时停止遍历
			if i >= len(prices) - 1:
				break
			if prices[i + 1] > prices[i]:
				sum = sum + (prices[i + 1] - prices[i])
				print(f'sum = sum + (prices[{i + 1}] - prices[{i}]) = {sum}')

			i += 1
		return sum

	def rotate(self, nums, k):
		"""
		:type nums: List[int]
		:type k: int
		:rtype: None Do not return anything, modify nums in-place instead.
		"""
		len_nums = len(nums)
		k = k % len_nums
		range((len_nums - 1) - k+1,len_nums, 1)
		for i in range(0,(len_nums-1)-k+1):
			nums.append(nums[i])
		del nums[0:(len_nums-1)-k+1]
		print(nums)


ll = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 9, 11, 20, 20]
prices = [11, 7, 8, 9, 3, 5, 7, 10, ]
rotate_ll = [1, 2, 3, 4, 5, 6, 7]
# [3]-[1] = 9-7=2,[7]-[4] =10-3=7,一共收入2+7=9
ss = Solution()
# result = ss.removeDuplicates(ll)
# prices_sum = ss.maxProfit(prices)
# print(f'prices_sum = {prices_sum}')
ss.rotate(rotate_ll,k=2)
