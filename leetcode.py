#Remove duplicates
#70%
i = 1
for b in range(len(nums) - 1):
    if nums[b] != nums[b + 1]:
        nums[i] = nums[b + 1]
        i += 1
return (i)

#Buy Time to Sell Stock

#10% faster
prof = []
for i in range(len(prices) - 1):
    prof.append(max(prices[i + 1] - prices[i], 0))
return (sum(prof))

#98% faster
prof = [0] * (len(prices) - 1)
for i in range(len(prices) - 1):
    prof[i] = (max(prices[i + 1] - prices[i], 0))
return (sum(prof)

#from right side
if not prices or len(prices) is 1:
    return 0
profit = 0
for i in range(1, len(prices)):
    if prices[i] > prices[i - 1]:
        profit += prices[i] - prices[i - 1]
    return profit

#Rotate Array
#80%
k = k % len(nums)
nums[:] = nums[-k:] + nums[:-k]

#Contains duplicate
#41%
#set takes O(n) and also takes O(n) space since its creating new array
return(len(set(nums))!=len(nums))

#Also O(n)  and O(n) space but doesnt use built in cheap functions
        d={}
        for i in nums:
            d[i]=d.get(i,0)+1

        for i in d.values():
            if i>1:
                return(True)
        return(False)

or

d = {}

for i in nums:
    if d.get(i, 0) != 0:
        return (True)
    else:
        d[i] = 1
return (False)
#Single Number
#72%
return(2*(sum(set(nums)))-sum(nums))
#dic array linear
d = {}
for i in nums:
    d[i] = d.get(i, 0) + 1

for i in d:
    if d[i] == 1:
        return (i)
#Intersection of Two Arrays
#Dict method 92% big o(n)
d = {}
ans = []
for i in nums2:
    d[i] = d.get(i, 0) + 1
for i in nums1:
    if d.get(i, 0) != 0:
        ans.append(i)
        d[i] -= 1
return (ans)

#two pointers method 74% , sorted makes slower
nums1, nums2 = sorted(nums1), sorted(nums2)
pt1 = pt2 = 0
res = []
while True:
    try:
        if nums1[pt1] > nums2[pt2]:
            pt2 += 1
        elif nums1[pt1] < nums2[pt2]:
            pt1 += 1
        else:
            res.append(nums1[pt1])
            pt1 += 1
            pt2 += 1
    except IndexError:
        break
return res

#alt way of 2 pters
        nums1.sort()
        nums2.sort()

        index_i, index_j = 0, 0
        result = []
        while index_i < len(nums1) and index_j < len(nums2):
        	if nums1[index_i] == nums2[index_j]:
        		result.append(nums1[index_i])
        		index_i += 1
        		index_j += 1
        	elif nums1[index_i] > nums2[index_j]:
        		index_j += 1
        	else:
        		index_i += 1
        return result
#Counter method 45%
counts = collections.Counter(nums1)
res = []
for num in nums2:
    if counts[num] > 0:
        res += num,
        counts[num] -= 1
return res

#PlusOne
#book method 60%
for i in range(1,1):
    print(i)
digits[-1] += 1

for i in reversed(range(1, len(digits))):
    if digits[i] != 10:
        break
    digits[i] = 0
    digits[i - 1] += 1

if digits[0] == 10:
    digits[0] = 1
    digits.append(0)
return (digits)

#Move Zeroes
#62%
pt = 0
for i in range(len(nums)):
    if nums[i] != 0:
        nums[pt], nums[i] = nums[i], nums[pt]
        pt += 1

#Two Sums
#46%
d={}
for i,n in enumerate(nums):
    des=target-n
    if des in d:
        return([d[des],i])
    else:
        d[n]=i
#99%
#linear time since using d.get is constantt tim
        d={}
        for i,n in enumerate(nums):
            sol=target-n
            if d.get(sol,-1)!=-1: 
                return([i,d[sol]])
            
            d[n]=i
#alternative way, bit slower
#store each numbers with index with its complement as a key value pair
        dic = {}
        for i, num in enumerate(nums):
            if num in dic:
                return [dic[num], i]
            else:
                dic[target - num] = i
#Valid Sudoku
def isUnitValid(self, unit):
    check = [x for x in unit if x != '.']
    return (len(set(check)) == len(check))


def isRowValid(self, board):
    for unit in board:
        if self.isUnitValid(unit) == False:
            return (False)
    return (True)


def isColValid(self, board):
    for unit in zip(*board):
        if self.isUnitValid(unit) == False:
            return (False)
    return (True)


def isSquareValid(self, board):
    for i in (0, 3, 6):
        for j in (0, 3, 6):
            unit = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if self.isUnitValid(unit) == False:
                return (False)
    return (True)


def isValidSudoku(self, board):
    return (self.isRowValid(board) and self.isColValid(board) and self.isSquareValid(board))
#Rotate Image
#54% if apply list, 98% if not
#reverse and then transpote
matrix[:]=map(list,zip(*matrix[::-1]))


#similar ans but doesnt use built in fct
#O(n**2) which if fastest time 0(1) since in place
        n = len(matrix[0])        
        # transpose matrix
        for i in range(n):
            for j in range(i, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i] 
        
        # reverse each row
        for i in range(n):
            matrix[i].reverse()

#Reverse String
#51% returns an iterator ready to traverse the list in reversed order
s[:]=reversed(s)
#93% does it inplace, modifies the list itself and does not return a value , only workson a list
s.reversed()
#15% does not change list but returns reversed slice
s[:]=s[::-1]
#reverse real way 2 pts 85%
#constant space, linear time
        left=0
        right=len(s)-1
        
        while left<right:
            s[left],s[right]=s[right],s[left]
            left+=1
            right-=1
        

#Reverse Integer
#96%
#linear time because of the reverse 
#constant space
sign = (x > 0) - (x < 0) #alt way of findiing sign  sign = [1,-1][x < 0] cus [1,-1] is a list and use 0 or 1 to index
nums = sign * int((str(abs(x))[::-1]))
if -(2) ** 31 <= nums < 2 ** 31:
    return (nums)
else:
    return (0)

#bit cleaner
        sign=(x>0)-(x<0)
        ans=int(str(abs(x))[::-1]) #have to use [::-1] because its a string not a list,and we dont want an iterator
        ans=sign*(ans)
        return(ans if -2**31<ans<2**31-1 else 0)

#First Unique Character in a String
#14% -> 65%
d = {}
seen = [] #change to set() since we knwo each value gon be unique
for i, n in enumerate(s):
    if n not in seen: #have to use seen so if letter shows up 3 times, doesnt add,delete,then add= only want to add each element once or else rmeove from dict if thers duplicates, keep new dict to onlyh ave single showne elements
        d[n] = i
        seen.append(n) #change to set if
    elif n in d:
        del d[n]
if len(d.values()) > 0: #change to just If d: brings from 65% to 87%
    return (min(d.values()))
else:
    return (-1)

#this way makes much faster by making seen a hash map for constant time search
        d={}
        seen={}
        for ind,num in enumerate(s):
            if num not in seen:
                d[num]=ind
                seen[num]=ind
            elif num in d:
                del d[num]
        
        return(-1 if len(d)==0 else min(d.values()))

#or this way faster time O(N) space O(N) since gotta make two N passses and save memory for hash map
90%
        count = collections.Counter(s)
        for i,n in enumerate(s):
            if count[n]==1:
                return(i)
        
        return(-1)

# much cleaner
d = {}
seen = set()
for i, n in enumerate(s):
    if n not in seen:
        d[n] = i
        seen.add(n)
    else:
        if n in d:
            del d[n]

if not d.values():
    return (-1)
else:
    return (min(d.values()))

#Valid Anagram
# 1 dic 70%, this one better than two dict becaues two dict takes 3n time and 2n space, while 1 dict takes 3nt time and 1n space
d = {}
for i in s:
    d[i] = d.get(i, 0) + 1

for i in t:
    if i in d:
        d[i] -= 1
    else:
        return (False)
for i in (d.values()):
    if i != 0:
        return (False)

return (True)

#sorted method 40%
return(sorted(s)==sorted(t))

#two dict method
d = {}
for i in s:
    d[i] = d.get(i, 0) + 1

d2 = {}
for i in t:
    d2[i] = d2.get(i, 0) + 1

if d == d2: #comparting two dicts is a recursive lookup, takes O(n) fro comparison
    return (True)
else:
    return (False)
#Valid Palindrome
#81% takes O(n) time  O(N) space because create temp array to hold the reverse slice array
s = s.lower()
s = [x for x in s if x.isalnum()]
return (s == s[::-1])

#takes 0(N) time and 0(1) space since just doing lookups/comparisons and moving pointers
        l=0
        r=len(s)-1
        
        while l<r: # has to go thru this check everytime after an event is triggered
            if not s[l].isalnum():
                l+=1
            elif not s[r].isalnum():
                r-=1
            else:
                if s[l].lower()==s[r].lower():
                    l+=1    
                    r-=1
                else:
                    return(False)
        return(True)

#40% takes big o constant in place
l, r = 0, len(s) - 1
while l < r:
    while l < r and not s[l].isalnum():
        l += 1
    while l < r and not s[r].isalnum():
        r -= 1
    if s[l].lower() != s[r].lower():
        return False
    l += 1;
    r -= 1
return True
# ez way
if not s:
    return (True)

s = [x.lower() for x in s if x.isalnum()]

l = 0
r = len(s) - 1
while l < r:
    if s[l] == s[r]:
        l += 1
        r -= 1
    else:
        return (False)
return (True)
# String to Integer
#49%
str = list(str.strip())
if len(str) == 0:
    return (0)

if str[0] == '-':
    sign = -1
else:
    sign = 1

if str[0] in ['-', '+']:
    del str[0]

res, i = 0, 0
while i < len(str) and str[i].isnumeric():
    res = res * 10 + int(str[i])
    i += 1

return (max(-2 ** 31, min(res * sign, 2 ** 31 - 1)))

#Implement strStr()
#80%, time complexity big o n * m tho during hay==needle chunk
if needle == '':
    return (0)
nl = len(needle)
for i in range(len(haystack) - nl + 1):
    if haystack[i:i + nl] == needle:
        return (i)
return (-1)

#Count and Say
def cns(str_):
    res = ''
    str_ += '#'
    c = 1
    for i in range(len(str_) - 1):
        if str_[i] == str_[i + 1]:
            c += 1
            continue
        else:
            res += str(c) + str_[i]
            c = 1

    return res


start = '1'
for i in range(n - 1):
    start = cns(start)
return start

#Longest Common Prefix
#73% linear time and constant space
if not strs:
    return ("")
shortest = min(strs, key=len)
for i, l in enumerate(shortest):
    for others in strs: #O(N here)
        if others[i] != l:
            return (shortest[:i]) # O(1) here since input doesnt affect lenght of shortest word
return (shortest)

#delete node in linked list
#36ms, 13mb ,97%
node.val = node.next.val
node.next = node.next.next

#remove n-th node from linked list
#20ms,12.7mb, 98%
fast = head
slow = head

for i in range(n):
    fast = fast.next

if fast == None:#this means that if fast already = None, it means the head is what we want to remove, otherwise we are trying to land on the node before the one we want to remove
    return (head.next)

while fast.next != None:
    fast = fast.next
    slow = slow.next

slow.next = slow.next.next
return (head)

#reverse node in linked list
#32 ms, 13.8mb , 97% O(L) time O(1) space, linear time cus gotta make a pass thru all # of nodes
prev = None
nextt = None
curr = head

while (curr != None):
    nextt = curr.next #need to save next node beacuse about to overwrite the curr.next to point to previous node(reversing direction)
    curr.next = prev
    prev = curr
    curr = nextt

return (prev)

#merge two sorted lists
#pointer method 36ms, 12.8mb, 94%
#O(n+m) so linear time and O(1) since only uisng a few pointers
dummy = cur = ListNode(0)
while l1 and l2:
    if l1.val < l2.val:
        cur.next = l1
        l1 = l1.next
    else:
        cur.next = l2
        l2 = l2.next
    cur = cur.next
cur.next = l1 or l2
return (dummy.next)

#palindrome linked list
#56 ms, 22.8mb 99%
#this takes O(N) time and O(1) since just using pointers
fast = slow = head
while fast and fast.next:#if # of nodes is even, then slow will land right at the node right after the midpoint since fast is traveling twice as fast, then you reverse the following linked list and compare it to the head list
    #goal is to get fast to the end (null) using 2x speed so just need to check fast and fast.next before advancin two nodes forward
    fast = fast.next.next # wanna go twice as fast as slow
    slow = slow.next

nd = None
while slow:#reverse the linked list at midway pt(if odd# of ndoes) right side of midway if even# of nodes
    nxt = slow.next
    slow.next = nd
    nd = slow
    slow = nxt

while nd: # compare the reverse linked list and head
    if nd.val != head.val:
        return (False)
    nd = nd.next
    head = head.next
return (True)

#alternative way 50%
#copies the linkedlist into an array then uses two pointers to find reverse and compare
#O(n) time for first pass to make the array and then later on for making reverse O(n) space cus making new array to hold linked list vals and also for reverse later on
        ans=[]
        cur=head
        while cur!=None:
            ans.append(cur.val)
            cur=cur.next
        
        l=0
        r=len(ans)-1
        og=ans.copy()
        while l<r:
            ans[l],ans[r]=ans[r],ans[l]
            l+=1
            r-=1
        return(og==ans)
#Linked list cycle
#48ms ,16.1 mb, 94%
#O(N) time cus it does one pass and O(1) cus we only use two pointers
if not head:
    return (False)

slow = head
fast = head.next # start off at head.next cus we already know head is not null and dont want to trigger comparison equals

while slow != fast: #this part check if there is a cycle
    if fast == None or fast.next == None: #this part check if there is no cycle
        return (False)
    fast = fast.next.next #if we cant prove either then, advance both forward until we prove one
    slow = slow.next
return (True)

#50% using hash tables 
# O(n) time  does one pass ,and O(n) space for creating the hash
        if not head:
            return(False)
        
        d={}
        while head!=None:
            if d.get(id(head),0)!=0:
                return(True)
            else:
                d[id(head)]=1
            head=head.next
        return(False)

#maximum depth of binary tree
#40ms ,13.8mb, 96%
#BFS , breadth first search
level = [root] if root else []
depth = 0

while len(level)>0:
    depth += 1
    queue = []
    for i in level:
        if i.left:
            queue.append(i.left)
        if i.right:
            queue.append(i.right)
    level = queue
return (depth)

#Depth first search, dfs
#dfs means it goes all the way down depth on left(if added last) then when it reaches its peak, goes bak down to next unresolve highest depth then digs thru that.
#40ms, 13.9mb,96%
depth = 0
stack = [(root, 1)] if root else []

while len(stack)>0:
    root, leng = stack.pop()

    if leng > depth:
        depth = leng

    if root.right:
        stack.append((root.right, leng + 1))
    if root.left:
        stack.append((root.left, leng + 1))

return (depth)

#Validate Binary Search Tree #just checking that each left element is less than its root val and right is greater
#alt way is to store inorder traversal of bt in a temp array and check if array is sorted in increasin order, cons r memory space and have to traverse an entire new array on top of traversin bt which we do for either methods, adding O(n) time


#RECURSION METHOD 44ms, 15 mb ,94% 
#O(N) space since we keep entire tree and O(N) time since we hit each node once
def isValidBST(self, root: TreeNode, floor=float('-inf'), ceiling=float('inf')) -> bool:
    if not root:
        return True
    if root.val >= ceiling or root.val <= floor : # right clause is for right subtree meaning values should never be lower than the root, left clause is for left subtree meaning values shud never be higher than the root. floor and ceiling in this case is the root node above the one bein compared
        return False
    # in the left branch, root is the new ceiling; contrarily root is the new floor in right branch
    return self.isValidBST(root.left, floor, root.val) and self.isValidBST(root.right, root.val, ceiling)

#without modifing inputs, create helper funct
    def isValidBST(self, root: TreeNode) -> bool:
        return(self.valBST(root,float('-inf'),float('inf')))
    
    def valBST(self,root,floor,ceiling):
        if not root:
            return(True)
        
        if root.val<=floor or root.val>=ceiling:
            return(False)
        
        return(self.valBST(root.left,floor,root.val) and self.valBST(root.right,root.val,ceiling))
            
            

#DFS method, 92%
#O(N) space since we keep entire tree and O(N) time since we hit each node once
        if not root:
            return(True)
        
        stack=[(root,float('-inf'),float('inf'))]
        
        while stack:
            root,floor,ceiling=stack.pop()
            if root.val<=floor or root.val>=ceiling:
                return(False)
            
            if root.right:
                stack.append((root.right,root.val,ceiling))
            if root.left:
                stack.append((root.left,floor,root.val))
        
        return(True)
#Symmetric Tree
#40 ms, 12.8mb,71%
#O(N) time because we traverse to each node once,
#O(N) space becus worst case we store all nodes into temp stack
if not root:
    return (True)

stack = [(root.left, root.right)]

while len(stack) > 0:
    left, right = stack.pop()
    if left is None and right is None:
        continue
    if left is None or right is None:
        return (False)

    if left.val == right.val:
        stack.append((left.left, right.right))
        stack.append((left.right, right.left))
    else:
        return (False)
return (True)

#Recursively
#O(n) time

return (self.helper(root.left, root.right) if root else True)


def helper(self, left, right):
    if not left and not right:
        return (True)
    if not left or not right:
        return (False)

    if left.val == right.val:
        outer = self.helper(left.left, right.right)
        inner = self.helper(left.right, right.left)

        return (outer and inner)
    else:
        return (False)


#Binary Tree Level Order Traversal
#28 ms, 13 mb,99%

if not root:
    return ([])

ans = []
level = [root]

while len(level) > 0:
    ans.append([node.val for node in level])
    temp = []
    for node in level:
        temp.extend([node.left, node.right])
    level = [leaf for leaf in temp if leaf]

return (ans)

# or
#O(N) time since we traverse to each node once
#O(N) space since the output has ups to N
#85%
if not root:
    return([])
level=[root]
ans=[]
while len(level)>0:
    res=[x.val for x in level]
    ans.append(res)
    queue=[]
    for i in level:
        if i.left:
            queue.append(i.left)
        if i.right:
            queue.append(i.right)
    level=queue
return(ans)
#sorted array to BST
#64 ms , 14.9mb, 98%
#find midpt, then split into two subtrees, repeat the same for both subtrees
#O(N) time since hit each node once
#O(N)  since need O(N) to keep output and O(Log N ) for recursion stack (its binary so log n), dont stack so just pick worst case
if not nums:
    return (None)

mid = len(nums) // 2 #takes the floor so 5//2= 2 or left middle

root = TreeNode(nums[mid])
root.left = self.sortedArrayToBST(nums[:mid])
root.right = self.sortedArrayToBST(nums[mid + 1:])

return (root)

#merge sorted array
#36 ms ,12.6mb, 93%
#2 pointers
#O(N) time cus its O(n+m) since makin pass thru both arrays
#O(1) space since we just doin swaps

while n > 0:
    if m == 0 or nums1[m - 1] < nums2[n - 1]: #edge case is when m==0 meaning we go thru all the elmenets in nums1 first, so cant compare anything to the remaining elmeents in nums2,
        #which case we just iterate thru remaining elemnts in nums2 until break outta n loop
        nums1[m + n - 1] = nums2[n - 1]
        n -= 1
    else:
        nums1[m + n - 1] = nums1[m - 1]
        m-=1

#first bad version
#binary search method
#28ms, 12.6mb, 90
#https://www.youtube.com/watch?v=SNDE-C86n88
#O(log n) since search space is halved each time
#O(1)
        left = 1
        right = n

        while i < j:
            mid = left + (right - left) // 2

            if isBadVersion(mid) == True:
                right = mid
            else:
                left = mid + 1

        return (left)

#brute force way 
#overflow error    
        res=None
        for i in range(1,n+1):
            if isBadVersion(i)==True:
                res=i
                break
        
        return(res)
[0]*0
#Climbing stairs
#the recurrence relation tells how many base cases there,
        #for bototm up, always gotta check edge case where only hte first base case exists, unless an empty array is doable, then gotta check both case caases
#78% bottom up , instead of waiting to make the computation calls when you need em, just compute the entire dp array right away, linear space
        # if n == 0:
        #     return (0) # dont need to include since problem states it wil be apositive integer
        if n == 1:
            return (1) #have to acocunt for base cases to make res[0]=1 and res[1] valid assignments, meaning account for when n= 0 or 1, then we can start iterationa at2
        res = [0] * n

        res[0] = 1
        res[1] = 2

        for i in range(2, n):
            res[i] = res[i - 1] + res[i - 2]

        return (res[-1])
#91%  Top down + memorization (list)  top down means ur making the computation calls as you need them / recursion/ might error in jupyter because of limit on amt of  recurcision calls
        if n not in self.dic:
            self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
        return self.dic[n]

    def __init__(self): # creates attributes within the instance of the class to call back later, can call it from other functions by referring it to by self within the funct
        self.dic = {1:1, 2:2}

#Best time to buy n sell
#97%, linear time, #basically keep global maximum going, only update if i-(i-1) turns a profit

        max_profit, min_price = 0, float('inf')
        
        for price in prices: #perform check to see if we shud use price as min price or to calc max profit or none
            min_price = min(min_price, price)#iterate thru and find the lowest price
            profit = price - min_price #keep trackin of profit but we only gon keep when the best one
            max_profit = max(max_profit, profit) #keep track of which day will give us the highest profit with respective to the lowest price we been tracking
        
        return max_profit

#Maximum Subarray
#50% linear time and constant space 
#Kadane's algorithm
        # dp=[0]*len(nums)
        # dp[0]=nums[0]
        
        for i in range(1,len(nums)):
            nums[i]=max(nums[i],nums[i-1]+nums[i])
        return(max(nums))

#little bit diff,instead of storing results in original array, keep updating maximum value contiously usin variable then return the var
        def maxSubArray(self, A):
            if not A:
                return 0

            curSum = maxSum = A[0]
            for num in A[1:]:
                curSum = max(num, curSum + num)
                maxSum = max(maxSum, curSum)

            return maxSum
#naive way would be bruteforce take the sum of all possible sub array 
#that would take n^2 time, n^3 if you compute each subarray from the start 

#House Robber
#87% bottom up
#linear time, linear space

        if not nums:
            return(0)
        if len(nums)==1:
            return(nums[0])
        dp=[0]*len(nums)
        dp[0]=nums[0]
        dp[1]=max(nums[0],nums[1])
        for i in range(2,len(dp)):
            dp[i]=max(nums[i]+dp[i-2],dp[i-1])
        return(dp[len(dp)-1])

#96 bottom up 
#linear time, constant space

        if len(nums)==0:#takes care of empty list
            return(0)
        if len(nums)==1: # takes care of only 1 house
            return(nums[0])
        nums[1]=max(nums[0],nums[1]) # takes caere of only 2 houses since the for loop wont do anything since range 2,2 makes nothin
        for i in range(2,len(nums)): # takes care of 3 houses + 
            nums[i]=max(nums[i-2]+nums[i],nums[i-1])
        
        return(max(nums))

#shuffle an array
#88%
    def __init__(self, nums: List[int]):
        self.arr=nums

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        return(self.arr)

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        new=random.sample(self.arr,len(self.arr))
        return(new)

#min stack
#17% linear time bad
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.arr=[]

    def push(self, x: int) -> None:
        self.arr.append(x)

    def pop(self) -> None:
        del(self.arr[-1])

    def top(self) -> int:
        return(self.arr[-1])

    def getMin(self) -> int:
        return(min(self.arr))

#75% constant time
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.arr=[]

    def push(self, x: int) -> None:
        currentmin=self.getMin()
        if currentmin==None or x<currentmin:
            currentmin=x
        self.arr.append((x,currentmin))

    def pop(self) -> None:
        self.arr.pop()

    def top(self) -> int:
        if not self.arr:
            return(None)
        
        return(self.arr[-1][0])

    def getMin(self) -> int:
        if not self.arr:
            return(None)
        
        return(self.arr[-1][1])

    #FizzBuzz
    #96% verbose
            ans=[]
        for i in range(1,n+1):
            if (i)%3 ==0 and (i)%5==0:
                ans.append('FizzBuzz')
                
            elif (i)%3==0:
                ans.append('Fizz')
            
            elif (i)%5==0:
                ans.append('Buzz')
            
            else:
                ans.append(str(i))
        return(ans)

    # 96% cleaner
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]# i % 5 == 0 and not 0 is 1
#choose whichever is present, if both are present then choose value on left

    #75%  alternate
        #O(N) time and O(1) space
            d={3:'Fizz',5:'Buzz'}
        ans=[]
        for z in range(1,n+1):
            word=''
            for i in d:
                if z%i==0:
                    word+=d[i]
            if not word:
                word=str(z)

            ans.append(word)
                
        return(ans)
    #count primes
    #88%
        if n<3:
        return(0)
    
    ans=[True]*n # keep it at n so we get spots for numbers 0,1,..n-1 (non-negative numbers less than n)
    ans[0]=False # base case, 0 is not prime
    ans[1]=False #base case 1 is not prime
    for i in range(2,int(n/2)+1): # start at 2 and iterate upwards, 2 and 3 is prime,not 4 tho.
        # just need to iterate up to the last # whose squared value is equal to or greater than n (greater than in case n isnt a perfect square root)(ex:lowest bound is 2*2=4)
        if ans[i]:
            ans[i*i:n:i]=[False]*len(ans[i*i:n:i]) 
            
    return(sum(ans))

    #Power of 3
    #78%
    #O(log3n) time
    #O(1)space
    if n<=0:
    return(False)
    
    while n%3==0:
        n=n/3
        
    return(n==1)
    #60% not faster but no loop or recursion
    return n > 0 and 1162261467 % n == 0

    #Roman to Int
    #55%
    #O(N) cus makes one pass
    #O(1) dict is gon be always same and amt is a constant
    dic={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    amt=0
    for i in range(len(s)-1):
        if dic[s[i]]<dic[s[i+1]]:
            amt-=dic[s[i]]
        else:
            amt+=dic[s[i]]
    
    return(amt+dic[s[-1]])

#Number of "1" Bits
#98% uses built in fct
return(bin(n).count('1'))

#71%
        c=0
        while n:
            n= n & n-1
            c+=1
            
        return(c)

#37% 
#recurcsion but not efficient

return 0 if n == 0 else 1 + self.hammingWeight(n&(n-1))

#Hamming Distance
#90% uses biult it bin funct

return(bin(x^y).count('1'))

#74%
# #1010
# XOR
# 1001
# =
# 0011
# x & x-1 is to remove the last bit
# x ^ y is XOR so it basically gives to you as a result the different bits.
        x = x ^ y
        y = 0
        while x:
            y += 1
            x = x & (x - 1)
        return y

#reverse bits

# x << y
# Returns x with the bits shifted to the left by y places 
#8%
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1)
            n>>=1
        return res

#97%
        res = 0
        for _ in range(32):
            res = res << 1 | (n&1)# changed this part
            n>>=1
        return res

#pascals triangle
# one way 42%
        res = [[1]]
        for i in range(1, numRows):
            res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
        return res[:numRows]
#more intuitive 82%
#adds the row to res array before modifying it
#O(numRows**2) time cus num of rows is how much elements we have to pass thru within each one
#O(numRows**2) gotta store each one
        lists = []
        for i in range(numRows):
            lists.append([1]*(i+1)) #i+1 is the # of elements for that row #
            if i>1 :
                for j in range(1,i):# so i is the index of last element and we want to exclude it since its gonna be always 1
                    lists[i][j]=lists[i-1][j-1]+lists[i-1][j]
        return(lists)
#method of modifying row first before adding to res array
#same time space as above
        res=[]
        for i in range(numRows):
            row=[1]*(i+1)
            if i>1:
                for j in range(1,i):
                    row[j]=res[i-1][j-1]+res[i-1][j]
            res.append(row)
        return(res)

#Valid parenthesis
#83% using stacks
#O(N) cus we just traverse thru the string once
#O(N) cus worst case we append all (((( which is proportional to n
        stack=[]
        dic={')':'(',']':'[','}':'{'}
        
        for i in s:
            if i in dic.values():# have to set closing  par as keys cus we comparing ( to ) , its not commutative where () is == )( since its iterating in order of s
                stack.append(i)
            elif i in dic.keys():
                if stack==[] or dic[i]!=stack.pop():
                    return(False)
            else:
                return(False)
        return(stack==[])
    
#missing number
#96% using constant space and constant time Gauss arithmetic sum formula
#actually takes O(N) time cus of the sum function, the gauss formula is constant tho
#proofs here that the n-th Triangular number, 1+2+3+...+n is n(n+1)/2
        n = len(nums)
        return int(n * (n+1) / 2 - sum(nums))

#83% using linear time and linear space
        tot=0               #array is [3,2,0] but shud be [3,2,1,0]<= this is range(4) [3,2,0] is missing one so add +1
        for i in range(len(nums)+1): # len(nums) gives us # of elements that shud be in the array( since it includes 0), gotta +1 so range can include em all
             # full array shud be 0-9 which is 10 values  but they only give us 9 out of those 10 values, so compute the full by taking range of len(n)+1 
            tot+=i
        return(tot-sum(nums))

#72%
#O(N) realy O(n+n) since making hash map takes linear time then doing one pas
#O(N) making the set
        d=set(nums)
        for i in range(len(nums)+1):
            if i not in d:
                return(i)

#3 Sum
# #88%
# Sorting takes O(NlogN)
# Now, we need to think as if the nums is really really big
# We iterate through the nums once, and each time we iterate the whole array again by a while loop
# So it is O(NlogN+N^2)~=O(N^2)

# For space complexity
# We didn't use extra space except the res
# So it is O(1).
        res=[]
        nums.sort()
        for i in range(len(nums)-2):# exclude the lsat two because we doin one pass and by the time we get to last two, theres no 3 elements to iterate thru
            l=i+1
            r=len(nums)-1
            
            if nums[i]>0: # its sorted so 1+ ? + ? any other two positive integers is not gonnnna equal to 0
                break
            if i>0 and nums[i]==nums[i-1]:#only doin one initial pass so dont need to iterate for same # agn
                continue
            
            while l<r:
                tot=nums[i]+nums[l]+nums[r]
                
                if tot<0:
                    l+=1
                elif tot>0:
                    r-=1
                else:
                    res.append([nums[i],nums[l],nums[r]])
                    while l<r and nums[l]==nums[l+1]:     #have to make sure we advance both pointers after adding them to results becus they arent gonna work together anymore since theres only one way it cud make the sum again which is with each other, but we dont want duplicates
                        l+=1
                    while l<r and nums[r]==nums[r-1]:
                        r-=1
                    l+=1
                    r-=1
        return(res)

        #Set Matrix Zeroes
        #65% 
        #O(N*M) TIME
        #O(1) SpACE

        row=len(matrix)
        col=len(matrix[0])
        iscol=None
        for i in range(row):
            if matrix[i][0]==0: # have to calc row 0 and col 0 separately because if there is a 0 in first row, we wud make the first value of the row 0 to idnicate it, however the same 0 would indicate the entire first column to be 0 even tho we dont know if there is a zero in the first column, its just based off the 0 that we found in the row
                
                iscol=True# scanning [x, , ]
                                    #[x, , ]
                                    #[x, , ]

            for j in range(1,col): # setting 0 at heads of row and col, gon use 0,0 to indicate for row 0 and not col 0
                if matrix[i][j]==0:
                    matrix[i][0]=0
                    matrix[0][j]=0
                          # scanning [,x,x ]
                                    #[,x,x ]
                                    #[,x,x ]
        
        for i in range(1,row):  #goin thru each value and lokin at its head col and row then filling in
            for j in range(1,col):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j]=0
                          # scanning [,  ,  ]
                                    #[, x, x]
                                    #[, x, x]
        if matrix[0][0]==0: # handling row 0 separeatly because the nums in row 0s will already be modified to 0 prob so its skewed when lookin up col and row for 0 check like above hcunk, have to use 0,0 as the indicator
            for i in range(col):
                matrix[0][i]=0
                          # scanning [x,x,x ]
                                    #[, , ]
                                    #[, , ]
        if iscol: #handling col 0 separeatly
            for i in range(row):
                matrix[i][0]=0
                          # scanning [x, , ]
                                    #[x, , ]
                                    #[x, , ]
#Group anagrams
#45%
#O(N*KlogK) time cus N is for the first pass of strs and KLogK is for sorting where K is the max length of largest str
#O(NK) to store values in hash map
        d={}
        for i in strs:
            d[tuple(sorted(i))]=d.get(tuple(sorted(i)),[]) +[i]
        return(list(d.values()))


#Faster way in O(N*K) time
#avoids sorting
        d={}
        
        for i in strs:
            arr=[0]*26
            for s in i:
                arr[ord(s)-ord('a')]+=1
        
            d[tuple(arr)]=d.get(tuple(arr),[])+[i]
        
        return(d.values())

#Longest substring without repeating characters
#92%
#O(N) time 
#O(N) space 
        maxlen=0
        start=0
        seen={}
        for i,n in enumerate(s):
            if n in seen and start<=seen[n]:
                start=seen[n]+1
            else:
                maxlen=max(maxlen,i-start+1)
            seen[n]=i
        return(maxlen)

#Longest Palindromic Substring
#60%
#O(N**2) 
#O(n) space worst case if the entire string is a palindrome then we storing it in the ans
        ans=''
        for i in range(len(s)):
            ans=max(ans,self.helper(s,i,i),self.helper(s,i,i+1), key=len)
        return((ans))
    
    def helper(self,s,l,r):
        while l>=0 and r<=len(s)-1 and s[l]==s[r]:
            l-=1
            r+=1
        
        return(s[l+1:r])

##method to save space, store answer as index instead of words
        def longestPalindrome(self, s: str) -> str:
            res = [0, 0]

            for i in range(len(s)):
                res = max(res, self.helper(s, i, i + 1), self.helper(s, i, i), key=lambda x: x[1] - x[0])
            return (s[res[0]:res[1]])

        def helper(self, s, i, j):
            while i >= 0 and j <= len(s) - 1 and s[i] == s[j]:
                i -= 1
                j += 1

            return ([i + 1, j])

        #INCREASING TRIPLET SUBSEQUENCE
#O(N) time and O(1) SPACE
        thresh1=float('inf')
        thresh2=float('inf')
        
        for i in nums:
            if i<=thresh1:
                thresh1=i
            elif i<=thresh2:
                thresh2=i
            else:
                return(True)
            
        return(False)


#ADD TWO NUMBERS

        head=dummy=ListNode(0)
        carry=0
        while l1 or l2 or carry:
            if l1:
                carry+=l1.val
                l1=l1.next
            if l2:
                carry+=l2.val
                l2=l2.next
            
            
            carry,rem=divmod(carry,10)
            dummy.next=ListNode(rem)
            dummy=dummy.next
            #or dummy.next=dummy=listnode(rm)
        return(head.next)
            
# ODD even linked list
# O(N) time O(1) SPACE
#94%
# def oddEvenList(self, head):
        headO=dummyO=ListNode(0)
        headE=dummyE=ListNode(0)
        
        while head:
            headO.next=head
            headE.next=head.next
            headO=headO.next
            headE=headE.next
            head=head.next.next if headE else None
        
        headO.next=dummyE.next
        
        return(dummyO.next)

#odd even without creating empty list nodes, so constant space

        if not head:
            return(None)
        if not head.next:
            return(head)
        oddH=dummyO=head
        evenH=dummyE=head.next
        head=head.next.next
        
        while head:
            dummyO.next=head
            dummyE.next=head.next
            dummyO=dummyO.nex
            dummyE=dummyE.next
            head=head.next.next if dummyE else None
        
        dummyO.next=evenH
        
        return(oddH)

# Intersection of two linked lists
# O(n+m) time, O(n) space to make hash map
# 73%
        d=set()
        while headA:
            d.add(headA)
            headA=headA.next
        
        while headB:
            if headB in d:
                return(headB)
            else:
                headB=headB.next
        return(None)

#O(n+m) time and O(1)space
#takes bit longer, but uses constant space!
#no infinite loop cus at the end, theyll intersect at null
#takes adv of fact that loopin thru everything will have them run into each other while theyre traversing each others list.
        if not headA or not headB:
            return(None)
    
        p1=headA
        p2=headB
        
        while p1!=p2:
            p1=headB if not p1 else p1.next
            p2=headA if not p2 else p2.next
        return(p1)


#Binary Tree Inorder Traversal
#O(n) time and space, would be O(d) space if didnt have to store all results in array
#Recursive method 66%
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        arr=[]
        self.helper(root,arr)
        return(arr)
    
    def helper(self,root,arr):
        if root:
            self.helper(root.left,arr)
            arr.append(root.val)
            self.helper(root.right,arr)
  
  #O(n)time and O(n)space

#Iterative function 25%
        res=[]
        stack=[]
        
        while stack or root:
            if root:
                stack.append(root)
                root=root.left
            else:
                temp=stack.pop()
                res.append(temp.val)
                root=temp.right
        return(res)

#Binary Tree Zig zag lever order traversal
#O(N) time O(N) space 
#mention how u cud use mod to reverse on odd levels 
        if not root:
            return([])
        res=[]
        flag=1
        queue=[root]
        while queue:
            temp=[x.val for x in queue]
            res.append(temp[::flag])            
            new=[]
            for x in queue:
                if x.left:new+= [x.left]
                if x.right: new+=[x.right]
            
            queue=new
            flag*=-1
            
        
        return(res)

#Construct Binary Tree from Preorder and Inorder Traversal        
#55%
#without using dequeu, bit slower
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if inorder:
            idx = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[idx])
            root.left = self.buildTree(preorder,inorder[:idx])
            root.right = self.buildTree(preorder,inorder[idx+1:])
            return root

#using deque, speeds up abit from collections import deque
#66%
class Solution:
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        preorder = deque(preorder)
        return self.getTree(preorder,inorder)
    
    def getTree(self,preorder,inorder):
        if inorder:
            idx = inorder.index(preorder.popleft())
            root = TreeNode(inorder[idx])
            root.left = self.getTree(preorder,inorder[:idx])
            root.right = self.getTree(preorder,inorder[idx+1:])
            return root


#iteratively
        if not preorder:
            return None
        
        root = TreeNode(preorder[0])
        stack = []
        stack.append(root)
        
        pre = 1
        ino = 0
        while (pre < len(preorder)):
            curr = TreeNode(preorder[pre])
            pre += 1
            prev = None
            while stack and stack[-1].val == inorder[ino]:
                prev = stack.pop()
                ino += 1
            if prev:
                prev.right = curr
            else:
                stack[-1].left = curr
                
            stack.append(curr)
        return root

#Populating Next Right Pointers in Each Node  
# 
#Recursion
# def connect1(self, root):
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        
        if root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
        
        self.connect(root.left)
        self.connect(root.right)
        
        return root    
#BFS methiod
        if not root:
            return(None)
        
        cur=root
        nex=cur.left
        
        while nex:
            cur.left.next=cur.right
            if cur.next:
                cur.right.next=cur.next.left
                # nex=cur.next.left
                cur=cur.next
            else:
                # cur.right.next=None
                cur=nex
                nex=cur.left
        return(root)
#Altenrate BFS
# BFS       
def connect2(self, root):
    if not root:
        return 
    queue = [root]
    while queue:
        curr = queue.pop(0)
        if curr.left and curr.right:
            curr.left.next = curr.right
            if curr.next:
                curr.right.next = curr.next.left
            queue.append(curr.left)
            queue.append(curr.right)
#DFS METHod
# DFS 
def connect(self, root):
    if not root:
        return 
    stack = [root]
    while stack:
        curr = stack.pop()
        if curr.left and curr.right:
            curr.left.next = curr.right
            if curr.next:
                curr.right.next = curr.next.left
            stack.append(curr.right)
            stack.append(curr.left)

#Kth Smallest Element in a BST
# recursive method 
# O(N) time to build traversal and O(N) space to then keep it
# 34%
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        return(self.inorder(root)[k-1])
    
    def inorder(self,root):
        return(self.inorder(root.left) +[root.val]+self.inorder(root.right) if root else [])        

#Number of islands
# Worst case runtime is O(mn) and space is O(mn)
        if not grid:
            return(0)
        count=0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    self.dfs(grid,i,j)
                    count+=1
        return(count)
    
    def dfs(self,grid,i,j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j]!='1':
            return(None)
        grid[i][j]='#'
        self.dfs(grid,i+1,j)
        self.dfs(grid,i-1,j)
        self.dfs(grid,i,j+1)
        self.dfs(grid,i,j-1)


# Letter COmbinations of a phone #
#time O(3**n x 4**m) where n is # of digits that map to 3 letters, and m is # of digits that map to 4 letters
#space is O(3**n x 4**m) as well to hold that many solutions
        d = {'2': ['a', 'b', 'c'],
             '3': ['d', 'e', 'f'],
             '4': ['g', 'h', 'i'],
             '5': ['j', 'k', 'l'],
             '6': ['m', 'n', 'o'],
             '7': ['p', 'q', 'r', 's'],
             '8': ['t', 'u', 'v'],
             '9': ['w', 'x', 'y', 'z']}

        output = []

        def backtrack(combo, digits):
            if len(digits) == 0:
                output.append(combo)
                return (None)
            else:
                for i in d[digits[0]]:
                    backtrack(combo + i, digits[1:])

        if digits:
            backtrack('', digits)

        return (output)


# Generate Parathensis



def generateParenthesis(self, n: int) -> List[str]:
    ans = []

    def backtrack(S='', left=0, right=0):
        if len(S) == 2 * n:
            ans.append(S)
            return (None)
        if left < n:
            backtrack(S + '(', left + 1, right)
        if right < left:
            backtrack(S + ')', left, right + 1)

    backtrack()
    return (ans)

# PERMUTATION
#dfs method

ans = []


def backtrack(combo, nums, ans):
    if not nums:
        ans.append(combo)
        return (None)

    for i in range(len(nums)):
        backtrack(combo + [nums[i]], nums[:i] + nums[i + 1:], ans)


    backtrack([], nums, ans)

return (ans)

#subset
Recursively
#O(N* 2**n) space and time to generate all subset and copy them into list
#Keep all subsets of length N  since each N element coudl be present or absent
res = []


def helper(path, index):
    res.append(path)
    #find all the paths that contain 1(1, 1-2,1-3,1-2-3), then all paths that contain 2 that dont include 1
    for i in range(index, len(nums)):
        helper(path + [nums[i]], i + 1)


helper([], 0)
return (res)


#brute force
#O(N* 2**n) space and time to generate all subset and copy them into list
#Keep all subsets of length N  since each N element coudl be present or absent
res = [[]]
for i in nums:
    res += [x + [i] for x in res]
return (res)

#Word search
#recursion
#O(n * m * len(word)) time complexity, because even if the cell doesnt work as the start of the word, we will access the cell agn to see if its 2nd or 3rd or 4th who knows
if not board:
    return (False)
row = len(board)
col = len(board[0])

for i in range(row):
    for j in range(col):
        if self.dfs(board, i, j, word):
            return (True)
return (False)


def dfs(self, board, i, j, word):
    if not len(word):
        return (True)
    if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or word[0] != board[i][j]:
        return (False)

    temp = board[i][j]
    board[i][j] = '#'

    res = self.dfs(board, i + 1, j, word[1:]) or self.dfs(board, i - 1, j, word[1:]) or self.dfs(board, i, j + 1,
                                                                                                 word[1:]) or self.dfs(
        board, i, j - 1, word[1:])
    board[i][j] = temp
    return (res)


# SORT COLORS

def sortColors(self, nums):
    red, white, blue = 0, 0, len(nums)-1
    
    while white <= blue: # have to less than or equal to because if the last element we check is a blue, we have to swap, but then check the swapped number again that is at the white pter.  by then the blue pter will be -1 and be equal to white pter
        #still need to run the check in case the swapped element is red
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            white += 1
            red += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1
# TOP K Frequent Elements
    d = {}
    for i in nums:
        d[i] = d.get(i, 0) + 1

    heaplist = []
    for i in d:
        heaplist.append((-d[i], i))

    heapq.heapify(heaplist)
    res = []
    for i in range(k):
        res.append(heapq.heappop(heaplist)[1])
    return (res)
#K-th Largest Element in an Array
    # O(k + (n - k)log(k)) time, min - heap
    import heapq
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        for i in range(len(nums)-k):
            heapq.heappop(nums)
        return(heapq.heappop(nums))

#FIND PEAK ELEMENT
#NAIVE APPROACH
#O(N) LIENEAR TIME
        nums.append(float('-inf'))
        for i,n, in enumerate(nums):
            if nums[i-1]<n and nums[i+1]<n:
                return(i)

#O(log n) BINARY SEARCH
        left = 0
        right = len(nums) - 1

        while left < right - 1:
            mid = (left + right) // 2

            if nums[mid] > nums[mid + 1] and nums[mid] > nums[mid - 1]:
                return (mid)
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            if nums[mid] < nums[mid - 1]:
                right = mid - 1

        return (left if nums[left] > nums[right] else right)

#Search for a Range
#O(log N) time 
#using bianar search
        def binaryleft(nums,target):
            left=0
            right=len(nums)-1
            while left<=right:
                mid=left+(right-left)//2
                if nums[mid]<target:
                    left=mid+1
                else:
                    right=mid-1
            return(left)
        
        def binaryright(nums,target):
            left=0
            right=len(nums)-1
            while left<=right:
                mid=left+(right-left)//2
                if nums[mid]>target:
                    right=mid-1
                else:
                    left=mid+1
            return(right)
        
        left,right=binaryleft(nums,target),binaryright(nums,target)
        return([left,right] if left<=right else [-1,-1])

#O(n) time 
#linear brute force

        for i in range(len(nums)):
            if nums[i]==target:
                left_ind=i
                break
        else:
            return([-1,-1])
        
        for i in range(len(nums)-1,-1,-1):
            if nums[i]==target:
                return([left_ind,i])

#Merge Intervals
# O(nlogn) because of sort, other than sort, we do a linear scar
#O(n) or O(1) space depending if we sort inplace or not, dont count output as space
        merged=[]
        
        intervals.sort(key=lambda x:x[0])
        
        for i in intervals:
            if not merged or merged[-1][1]<i[0]:
                merged.append(i)
            else:
                merged[-1][1]=max(merged[-1][1], i[1])
        
        return(merged)

#sorting manually
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                if intervals[i][0] > intervals[j][0]:
                    temp = intervals[i]
                    intervals[i] = intervals[j]
                    intervals[j] = temp

        res = []

        for i in intervals:
            if not res or i[0] > res[-1][1]:
                res.append(i)
            else:
                if i[1] > res[-1][1]:
                    res[-1][1] = i[1]
        return (res)
#Search IN A ROTATED SEARCH ARRAY
#O(Log n) time
        if not nums:
            return(-1)
        
        left=0
        right=len(nums)-1
        
        while left<=right:
            mid=left+(right-left)//2
            if nums[mid]==target:
                return(mid)
            
            if nums[left]<=nums[mid]:# for the case that the nums is [3,4,5,6,mid,7,1,2], so we can do a target check on the sorted left subarray, have to use equal to sign in case theres two elements [1,2] and we used floor divide to find midpt
                if nums[left]<=target<nums[mid]:
                    right=mid-1
                else:
                    left=mid+1
            else:#for the case that the nums is [6,7,1,2,mid,3,4,5] so we do the target check on a sorted array  on right side.
                if nums[mid]<target<=nums[right]:
                    left=mid+1
                else:
                    right=mid-1
        
        return(-1)
#O(n) naive linear time method
        if not nums:
            return(-1)
        
        for i in range(len(nums)):
            if nums[i]==target:
                return(i)
        return(-1)        
    
#Search a 2d Matrix
#brute force O(m*n) time
        if not matrix:
            return(False)
        row=len(matrix)
        col=len(matrix[0])
        for i in range(row):
            for j in range(col):
                if matrix[i][j]==target:
                    return(True)
        
        return(False)

#Linear O(m+n)
#Constant space
        if matrix:
            row=len(matrix)-1
            col=0
            width=len(matrix[0])

            while row>=0 and col<width:
                if matrix[row][col]==target:
                    return(True)
                elif matrix[row][col]>target:
                    row-=1
                else:
                    col+=1
        
        return(False)

#DIVIDE TWO INTEGERS
#O(logN)
        sign=(dividend>0) == (divisor>0) # get sign
        
        a,b,res=abs(dividend),abs(divisor),0
        
        while a>=b:         #durin last round, it won't enter inner wihle loop, just keeps x at 0
            x=0     
            while a>=b<<(x+1): # x dictates how many times (3*2) can go into A(10),b<<1 = 3*2
                x+=1            #first round , x=1 because 3*2 goes into 10 but not 3*2*2
            res+=1<<x           #keep track of how many times 3 goes into 10, 1<<1 = 2, 1<<0=1
            a-=b<<x             #subtract 3*2 from 10= 4 ,want to leave at least one more round
                            #at very end, 4-3 =1 so 1 is no longer greater than 3 and while loop breaks
        return(min(res if sign else -res, 2**31-1)) # min to prevent integer overflow


#Pow (x,n)
#RECURSIVE METHOD
        if not n:
            return(1)
        if n<0:
            return(1/self.myPow(x,-n)) #have to convert negative exponent to positive 
        if n%2:
            return(x*self.myPow(x,n-1))
        return(self.myPow(x*x,n/2))

        #ex: 4**2 => 16**1 => 1
#iterative method
        if n<0:
            x=1/x
            n=-n
        pow=1
        while n:
            if n & 1: #if its odd exponent, multiply it into ans right away, and then resume dividing exponent by half and squaring the # , last exponenet number will eventually be one which then breaks outta loop after its shifted
                pow*=x
            x*=x        #otherwise keep storing multiplting squared versions in x
            n>>=1
        return(pow)

        # Shifting all the bits 1 to the right (m >>= 1) has the effect of cutting an integer in half and dropping the remainder. The bitwise and (m & 1) basically compares the last bit in m with the single bit in 1, and if they're both 1 
        # then it evaluates to 1 (otherwise it evaluates to 0). So odd m will evaluate to 1 (or True), and even m will evaluate to 0 (or False).

#O(n) MAXIMUM SUBARRAY
# 
        maxsum=cursum=nums[0]
        
        for i in range(1,len(nums)):
            cursum=max(nums[i],cursum+nums[i])
            maxsum=max(maxsum,cursum)
        return(maxsum)
        
        
#SPIRAL MATRIX
# 
        if (len(matrix) == 0):
            return []
        
        new_matrix = []
        for j in range(len(matrix[0])-1, -1, -1):     # for each column starting from end, get the column starting at 2nd elmenet downwarrds
            new_lst = []
            for i in range(1, len(matrix), 1):
                new_lst.append(matrix[i][j])  #add the col to be a new row in new mat
            new_matrix.append(new_lst) #only add the new columns into new mat to be used as input in next recursion
            
        return matrix[0] + self.spiralOrder(new_matrix)        #concatenate lists
         # basically want to get the column on the right first, 
         # 1 2 3    1 2 3        6 9 
         # 4 5 6    4 5 x        5 8
         # 7 8 9    7 8 x        4 7

#Sqrt(x)
        #use binary search
        left=0
        right=x
        while left<=right:
            mid=left+(right-left)//2 #find midpt while avoiding int overflow
            if mid*mid<=x<(mid+1)*(mid+1): #have to check if the x is between mid**2 and mid+1**2 because x may not have a perfect squuare
                return(mid) #it is < for mid+1 because if its equal, then u wud return (mid+1) not mid
            
            if mid*mid<x:  #if squared
                left=mid+1
            else:
                right=mid
#LONGEST INCREASING SUBSEQUENCE
#O(n**2) quadratic time and O(n) linear space

        if not nums:
            return (0)
        res = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i] and res[i] < 1 + res[j]:
                    res[i] = 1 + res[j]
        return (max(res))

#COIN CHANGE
#O(A*C) time where A is amount and C is # of coins
#O(A) space complexity to store minimum # of coins to make up each increment of amount

        res = [0] + [float('inf') for x in range(1, amount + 1)] #leave first value as 0 for basecase since no coins needed to make 0$

        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0: #if amount - coin_amt is > 0, find least coins to make up remainder, else it will be float('inf')
                    res[i] = min(res[i], res[i - coin] + 1) #choosing the least amount of coins to make up specific i,

        if res[-1] == float('inf'):
            return (-1)
        else:
            return (res[-1])

# JUMP GAME
#O(n) Linear time
#could be constant space but below is linear space cus of extra bottom up array
        d = [False] * len(nums) #exlcude for const
        d[-1] = True #exclude for const
        rightmost = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= rightmost:
                d[i] = True #exclude
                rightmost = i

        return (d[0] == True) #use rightmost==0 for const


#Unique Paths
#O(n*m) time to go thru all cells and update
#O(n*m) space to create 2d matrix
        row=n #relabeling rows and col in a way makes more sense
        col=m
        res=[[1 for x in range(col)] for x in range(row)]

        for i in range(1,row):
            for j in range(1,col):
                res[i][j]=res[i][j-1]+res[i-1][j] # in 2d matrix, only need to sum boxes to left and above it to current box to calc all possible ways of gettin to that box
        return(res[-1][-1])

#Method that uses less space by making 1d array instead of mat
#O(n*m) time
#O(n) space
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] = dp[j - 1] + dp[j]
        return dp[-1] if m and n else 0

#DAILY TEMPERATURE
#O(n) slow linear, its actually O(n*w) where W is # of temperatures allowed which is 71
        nxt = [float('inf')] * 102 # array to store min index of all temps
        ans = [0] * len(T)
        for i in xrange(len(T) - 1, -1, -1): #going bakwards in order to fill out array with index of temps
            #finding the mininmum index of all the temperatures greater than current one
            warmer_index = min(nxt[t] for t in xrange(T[i]+1, 102))
            if warmer_index < float('inf'):
                ans[i] = warmer_index - i
            nxt[T[i]] = i #also goin bakwards so the index saved in index array will always the minimum for that temp
        return ans

#O(n) linear faster, uses stack and doesnt have to go thru every large ranges of temperature, just compares the given temperatures
#general idea is to keep adding indexes to stack and remove the ones that have lower temp than the current one bbecuz current does everythign better (lower index and higher temp)
        ans = [0] * len(T)
        stack = [] #indexes from hottest to coldest
        for i in xrange(len(T) - 1, -1, -1): #goin bakwards so each element we review is gonna have least amt of days so only need to compare temp, also need to fill in stack
            while stack and T[i] >= T[stack[-1]]: #going thru stack and discarding any value that has a lower temp, becus the current value can replaec it since it got a smaller ind and higher temp
                stack.pop()
            if stack: #once u run into a vaule in stack that has higher temp, then calc days away and put it into answer array
                ans[i] = stack[-1] - i
            stack.append(i) #if u broke out of the while loop cus stack ran out , then just add the current ind
        return ans

#PRODUCT OF ARRAY EXCEPT SELF
#O(N) linear time tho we do it 3 times
#O(N) space tho we do it twice for left and right (output dont ocunt)
        L = [0] * len(nums) #L[i] represents the product of all elements left of i
        R = [0] * len(nums)
        res = [0] * len(nums)
        L[0] = 1 #for first index, nothing to left so have to initialize with just 1 so product stays the same
        for i in range(1, len(nums)):
            L[i] = L[i - 1] * nums[i - 1] #to get product of all ements left of i, have to use product of previous (product of left elements) and the num i so  1,2,3 => x,x,3 where 2 ist he num and 1 is the product of left  of 2

        R[len(nums) - 1] = 1
        for i in reversed(range(len(nums) - 1)):
            R[i] = R[i + 1] * nums[i + 1]

        for i in range(len(nums)):
            res[i] = L[i] * R[i]
        return (res)

#MAXIMUM PRODUCT SUBARRAY
#O(N)
        cur_max = nums[0]
        cur_min = nums[0]
        res = nums[0]
        prev_max = nums[0]
        prev_min = nums[0]

        for i in range(1, len(nums)):
            cur_max = max(prev_max * nums[i], prev_min * nums[i], nums[i]) # have to store min value in case a negative number comes up , then the min num will be a negative if different and can offset it, makin it the greatest prdouct
            cur_min = min(prev_max * nums[i], prev_min * nums[i], nums[i]) #can assign them both simultaneously which eliminates need for prev_max/prev_min
            res = max(res, cur_max) #also have to ues nums[i] by itself in case there is a 0, then for the number following the 0, max and min will be 0 * num so have to account for just using num
            prev_max = cur_max
            prev_min = cur_min

        return (res)
    
#FIND MINIMUM IN ROTATED SEARCH ARRAY
#O(log n) for binary search
#O(1) space
        
        left = 0
        right = len(nums) - 1
        if not nums: #dont needa check this
            return (None)
        if len(nums) == 1:
            return (nums[0])
        if nums[right] > nums[0]:
            return (nums[0])

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:  # basically needa check for inflection point, if not then move boundaries accordingly
                return (nums[mid + 1])

            if nums[mid] < nums[mid - 1]:
                return (nums[mid])

            if nums[mid] > nums[0]:
                left = mid + 1
            else:
                right = mid - 1

        #CONTAINER WITH MOST WATER
#O(N) linear single pass
#O(1) constant space
        left = 0
        right = len(height) - 1
        water = 0

        while left <= right:
            water = max(water, min(height[left], height[right]) * (right - left)) #calculate area of widest container using shortest height and width diff
            if height[left] < height[right]: # shrink the shortest height, becuz if u shrink the taller one then ur area is only going to decrease (cant even equal to it cus width reduced by 1 so area will be guaranteed lower)
                left += 1 # at least if u shrink shorters, theres a chance it will shrink to a taller wall that wil compsenate for the smalller width
            else:
                right -= 1

        return (water)


#Flatten a Multilevel Doubly Linked List
#O(n) linear time cus hit each node once
#O(n) space ,using explciit call stack which worst case can contian all nodes, explciit better cus its way bigger than memory stack
        if not head:
            return (None)

        stack = [head]
        prev = Node(0) #dummy node for the sake of while loop function

        while stack:
            temp = stack.pop()
            temp.prev = prev
            prev.next = temp
            prev = temp
            if prev.next: #add neighboring node to process after finish processing all nodes on child level
                stack.append(prev.next)
            if prev.child: #add child node last to process it first
                stack.append(prev.child)
                prev.child = None
        head.prev = None #head has to have no prev pointer
        return (head)

#Recursive Method

    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return (None)
        self.travel(head)
        return (head)

    def travel(self, cur):
        while cur: #pretty much just using next_node to connect child node back to current level, then have tail
            next_node = cur.next
            if not next_node:  #have to get next_node so we know whether it exists to assign its prev pointer to childtail
                #assign tail so when level reachs end, the tail node can be returned to the original parent node to connect to the next node in prior level
                tail = cur  # reached the last node in current level, assign it to 'tail' for return
            if cur.child: #if it contains a child node, assign pointers and recursively traverse the child node's level
                cur.child.prev = cur
                cur.next = cur.child
                child_tail = self.travel(cur.child) #returns tail for the child node's level
                if next_node:  # if there exists a node in the prior level, connect its prev pointer to the child node's tail. if there is none, then no need
                    next_node.prev = child_tail
                child_tail.next = next_node  # have to connect child_tail back to prior level even if it is null
                cur.child = None  # clearing child pointers
            cur = cur.next  # will either continue traversing current level or break out of loop if cur was last node in level

        return (tail)  # returns tail node of level

# INVALID TRANASACTIONS
#O(n**2) Brute force
        invalid = []
        for i in range(len(transactions)):
            name, time, cost, city = transactions[i].split(',')
            if int(cost) > 1000:
                invalid.append(transactions[i])
            #cant use continue here because if a future trans has a conflicting time with the curent trans, we wont be able to detect because our second loop only looks at trans after it
            for j in range(i + 1, len(transactions)):
                name1, time1, cost2, city2 = transactions[j].split(',')
                if name1 == name and city2 != city and abs(int(time) - int(time1)) <= 60:
                    invalid.append(transactions[i])
                    invalid.append(transactions[j])

        return (list(set(invalid)))


#O(nlogn + 2n) time faster
        d={}
        invalid=[]
        transactions.sort(key=lambda x:int(x.split(',')[1]))
        for i in range(len(transactions)):
            name,time,cost,city=transactions[i].split(',')
            d[name]=d.get(name,[]) +[[name,time,cost,city]]

        for name,trans_arr in d.items():
            left=right=0

            for trans in trans_arr:
                name1,time1,cost1,city1=trans
                if int(cost1)>1000:
                    invalid.append(','.join(trans))
                    continue # dont wanna add duplicates
#use left and right pointers to set boundaries +60/-60 min for certain transaction by certain person
#then iterate through all the trans in those boundaries to see if city mismatch, if so, then add the original trans, when the other matches come up in iterator theyll get added, but we do this to avoid dup
                while left<len(trans_arr)-1 and int(trans_arr[left][1])<int(time1)-60:
                    left+=1
                while right<len(trans_arr)-1 and int(trans_arr[right+1][1])<=int(time1)+60: #inclusive, when we find first right # that is over 60 bound, its using right+1 ind so we want all #s up to right ind including right
                    right+=1
                for poss_invalid in trans_arr[left:right+1]:  #use right+1 becuz we wanna include trans at the right index
                    if poss_invalid[3]!=city1:
                        res.append(','.join(trans))
                        break
        return(invalid)

#ALL PATHS FROM SOURCE TO TARGET
#O(2**n * N**2) exponential times quadratic time
#there exists two possibilities for each node (except for the first and the last): appearing or not appearing in the path. Therefore, we have 1*2*...*2*1 = 2^(N-2) possibilities for paths.
#O(2**n *N) space
#BREADTH FIRST SEARCH
        if not graph:
            return (None)

        res = []
        queue = [[0]]

        while queue:
            temp = []
            for path in queue:
                if path[-1] == len(graph) - 1:
                    res.append(path)
                    continue

                for i in graph[path[-1]]:
                    temp.append(path + [i])

            queue = temp

        return (res)

#DEPTH FIRST SSEARCH
#O(2**N * N) time complexity, takes O(N) time to collect a path and then theres 2**N posssible comb
#O(2**N *N) space
        if not graph:
            return (None)

        res = []
        stack = [[0]]

        while stack:

            path = stack.pop()
            if path[-1] == len(graph) - 1:
                res.append(path)
                continue

            for edge in graph[path[-1]]:
                stack.append(path + [edge])

        return (res)

#RECURSION/BACKTARACKING
#O(2**n * n**2)
#O(2**n) comes from there exists two possibilities for each node (except for the first and the last): appearing or not appearing in the path. Therefore, we have 1*2*...*2*1 = 2^(N-2) possibilities for paths.
#O(n**2) comes from  iterating thru all nodes reachable from A and then for each node B, iterate thru every node reachable from B, when calling "for path in" its collecting paths  from the end incremenetally and merging em together
# O(2**n *N) space
        if not graph:
            return (None)

        def solve(node):
            if node == len(graph) - 1: return ([[node]]) #have to put node in nestd array because "for path in solve(edge)" selects [node] so we can then merge the lists

            ans = []
            #ex:[0,1,2]
            for edge in graph[node]:  # graph[node] = [1]
                for path in solve(edge): # solve(1) calls solve(2) which returns [[2]], so for solve(1) - it ans.appends([1]+[2])  then its return its ans=[[1,2],[1,1.5,2]]
                    ans.append([node] + path) # for solve(0) , for each path in ans=[[1,2]] , append to original ans which will have master [[1,2],[2,3]]
            return (ans)

        return (solve(0))

# TWO CITY SCHEDULING
#O(nlogn)
#O(n) linear space to make new arrays containing costs for the two grups
        costs.sort(key=lambda x: x[0] - x[1])
        sumA = sum([cost[0] for cost in costs[:len(costs) // 2]])
        sumB = sum([cost[1] for cost in costs[len(costs) // 2:]])

        return (sumA + sumB)

#O(nlogn)
#O(1) constant space since we just incremnting a variable
        costs.sort(key=lambda x: x[0] - x[1])

        sumA = 0
        sumB = 0

        for i in costs[:len(costs) // 2]:
            sumA += i[0]

        for i in costs[len(costs) // 2:]:
            sumB += i[1]

        return (sumA + sumB)

#O(n**2)
#manually sorting in brute force way ( two pointers)
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):
                diff1 = costs[i][0] - costs[i][1]
                diff2 = costs[j][0] - costs[j][1]

                if diff2 < diff1:
                    costs[i], costs[j] = costs[j], costs[i]

        a = sum([x[0] for x in costs[:len(costs) // 2]])
        b = sum([x[1] for x in costs[len(costs) // 2:]])
        return (a + b)

# DECODE STRING
#O(N) linear time
#O(N) space for the stack
        stack = []
        curString = ''
        curNum = 0

        for i in s:
            if i.isdigit():  # in case theres mulitple digits
                curNum = curNum * 10 + int(i)
            elif i == '[':
                stack.append(
                    curString)  # have to append curString here because there may be mulitiple digits which would add dupplicates
                stack.append(curNum)
                curString = ''
                curNum = 0  # have to append here as well in case theres a nested bracket
            elif i == ']':  # marks the end of the segment whose # is most recent in stack and its word is the curString
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num * curString

            else:
                curString += i  # just adding letters to cur string

        return (curString)
#REMOVE ALL ADJACENT DUPLICTES 2 #1209
#O(N) time
#O(N) for stack

        stack = [['#', 0]]
        for c in s:
            if stack[-1][0] == c: #traverse and add tuple of letter with a count of 1,
                stack[-1][1] += 1 #when encounter new letter, check if most recent stack is same char, if so then increment cnt, tryng to get k in a row
                if stack[-1][1] == k: #if count is == k , then pop out tuple
                    stack.pop()
            else:
                stack.append([c, 1]) #otherwise just add tuple

        #reconstruct word here by iterating thru stack
        word = ''
        for letter in stack:
            word += letter[0] * letter[1]
        return (word)


#Design Undergorund System
    def __init__(self):
        self.d = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        # store start station and check-in time using id as key
        self.d[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        # calculate trip duration and then store it in dictionary using a concatenation of the start/end station as the key
        duration = t - self.d[id][1]
        self.d[self.d[id][0] + stationName] = self.d.get(self.d[id][0] + stationName, []) + [duration]

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        # calc average of desired route
        return (sum(self.d[startStation + endStation]) / len(self.d[startStation + endStation]))

#LRU CACHE
#All operations are in O(1) constant time
class Node:
    def __init__(self, x, y):
        self.key = x
        self.val = y
        self.next = None
        self.prev = None


class LRUCache:

    def __init__(self, capacity: int):
        self.size = capacity
        self.d = {}
        self.head = Node(0, 0) #head and tail will be dummy nodes so dont have to worry about edge cases
        self.tail = Node(0, 0)
        self.head.next = self.tail #head.next will be LRU
        self.tail.prev = self.head #tail.prev will be newest additions

    def get(self, key: int) -> int:
        if key in self.d:
            self._remove(self.d[key]) #have to refresh its position in cahce
            self._add(self.d[key])
            return (self.d[key].val)
        else:
            return (-1)

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self._remove(self.d[key]) #tho were just updating a current cache object,remove and add it with its new value to refresh its position in cache

        elif len(self.d) == self.size: #before we can add new cache value, have to remove LRU one.
            lru = self.head.next
            self._remove(lru)
            del self.d[lru.key] #the dictionary keeps track of key/values that are actually stored on the cache. Once we remove a node from the cache, we have to remove its dictionary reference as well.
            # This is so the get() function works properly and returns a -1 when trying to retrieve a value thats not on the cache anymore.

        self._add(Node(key, value))

    def _remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _add(self, node): #always going to be adding elements right before tail since thats where the most recent used elements are
        self.d[node.key] = node
        p = self.tail.prev
        p.next = node
        node.prev = p
        node.next = self.tail
        self.tail.prev = node


#Word Break
#O(n**2) time complexity cus of nested loop
#O(n) space to contain res array
        res = [False] * (len(s) + 1)  # extra spot at end to check that entire previous string is valid
        # res[i] means letters in s[:i] can be decomposed into segments
        # so just need to find letters in s[i:] that make up word
        res[0] = True  # base case to get program going,'' is valid so it works

        for i in range(len(s)):  # move thru i, in case theres a better first word to choose from that works out
            if res[i]:
                for j in range(i + 1, len(s) + 1):  # when j=len(s), it works out for the s check becuz of indexing
                    if s[i:j] in wordDict: #edge case; this would be s[i:len(s)] which is compltely valid since it is testing the rest of the string
                        res[j] = True  # will asssign True for all spots where words could be formed, have to find the corect combination that will take up all string
        return (res[-1])  # indicator if entire previous string is valid

#Insert Delete GetRandom O(1)
#O(1) for all operations, not sure about random

import random

class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums=[]
        self.d={} #need hashmap to keep track of position so we can remove in constant time

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val not in self.d:
            self.nums.append(val) #add to end
            self.d[val]=len(self.nums)-1
            return(True)
        else:
            return(False)

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.d:# take last element and use it to override the val to be removed
            ind=self.d[val] #extract ind to override
            last=self.nums[-1] #extrat last val
            self.nums[ind]=last #override val
            self.d[last]=ind #update dictionary index for val
            del self.d[val] #update dict of removal of original val
            del self.nums[-1] #remove last element from nums since we moved it
            return(True)
        else:
            return(False)

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return(self.nums[random.randint(0,len(self.nums)-1)]) #random

#ADD TWO NUMBERS II
#O(max(n,m) time
#O(max(n,m)) linear space to store stacks / create new nodes for anaswer
        stack1, stack2 = [], []

        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next

        carry = 0
        dummy = head = ListNode(0)

        while stack1 or stack2 or carry:
            if stack1:
                carry += stack1.pop()
            if stack2:
                carry += stack2.pop()

            remainder = carry % 10

            # we want the head of our answer to be the most sig digits, so each new number we calc has to be added at the head
            # assign the current list to temp pointer , head is just a placeholder so we use next
            temp = head.next

            # add new node at the head
            head.next = ListNode(remainder)

            # reattach rest of the linked list to new head
            head.next.next = temp

            carry = carry // 10

        return (dummy.next)

#RECURSIVE METHOD
#SUPER COMPLEX
#O(n) time
#O(1) space cus we just updating L1

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # get lengths of both lists so we can make adjustments if ones longer
        len1, len2, carry = self.getLen(l1), self.getLen(l2), 0

        # set the longer linkedlist to l1 (need to be consistent)
        if len1 < len2:
            l1, len1, l2, len2 = l2, len2, l1, len1

        # begin recursion to calc carry and update l1.val to be remainders for each digit sum (this is why we made l1 be the longer list)
        carry = self.recur(l1, len1, l2, len2)
        # if there is a carry at the very end, then we create a new node for it and connect it to l1 which will be a linkedlist containing all of our computed values
        if carry:
            head, head.next = ListNode(carry), l1
            return head
        # if there is no carry for most sig digit of l1,then just return l1 which is already linked to all the least sig digits computations.
        return l1

    def recur(self, l1, len1, l2, len2):
        # if reached beginning(least sig digits)
        if not (l1 or l2): return 0

        # compare lens to see if we should adjust
        # if len1 is bigger, we traverse down l1 but keep l2 the same until lens are equal
        if len1 > len2:
            # since lengths are different, l1 will run out of digits from l2 to add,the sum for l1 digits that dont have a l2 counterpart is just l1.val + carry
            l2v = 0
            carry = self.recur(l1.next, len1 - 1, l2, len2)
        else:
            l2v = l2.val
            carry = self.recur(l1.next, len1, l2.next, len2)

        # modify l1.val to hold remainder for each sum iteration
        l1v = l1.val
        l1.val = (l1v + l2v + carry) % 10

        # return carry, carry may equal 0 at the very end so we'll return l1 in that case
        return (l1v + l2v + carry) // 10

    def getLen(self, node):
        l = 0
        while node:
            node, l = node.next, l + 1
        return l


#VALID TRIANGLE NUMBER
#similar to 3 sum, iterate thru and use 2 pointers that traverse inwards
#O(n**2) cus of nested loop (linear scan), sort is O(nlogn) too
#O(log N) sorting takes log N space, otherwise itd be constant, just updating res value and 2 pters
        res = 0
        nums.sort()
        #or use  for i in reversed(range(2,len(nums))):
        for i in range(len(nums) - 1, 1, -1): #approach is to set target as the longest side, then find 2 smaller that adds up, cant do normal range cuz [1,2,3] 2 and 3 are already greater than 1 on their own
            #cud prol iterate normal range, but equation wud be diff like nums[i]+nums[left]> nums[right], and move left index forward if its less than the right
            left = 0
            right = i - 1

            while left < right:
                if nums[left] + nums[right] > nums[i]: #its valid triangle if sum of 2 sides is greater than the 3rd
                    res += right - left #if left index works, then any index plus the right will work since its sorted so left # only gon increase
                    right -= 1 #move right one in after iteration thru with initial right, get all combinations
                else:
                    left += 1 #if num[left] and num [right]] werent big enuf, right is already max for that iteration so need to move left

        return (res)

#FIRST MISSING POSITIVE
#O(N) LINEAR TIME
#O(1) CONSTANT SPACE
# if len(array) is 5, need to have indexes 0,1,2,3,4,5  to eliminate one by one, append a 0 to get to 6 indexes with the last index being the 5th
#length is now 6
        nums.append(0) #adding a placeholder for sake of having an index that correspondgs to a possible smallest positive #
        n = len(nums)
        for i in range(len(nums)): #delete those useless elements
            if nums[i]<0 or nums[i]>=n: #throw out any number greater or equal to 6, so when we use mod on a small positive #,itll return the same # , dont want negative # nor do we have neg ind, so get rid of emt oo
                nums[i]=0 #just set it to 0 so itll increment our dummy 0 index in the next step, 0%6= 0
        for i in range(len(nums)): #use the index as the hash to record the frequency of each number
            # if any of  0,1,2,3,4,5 exists, then that number mod 6 will return the exact same #, we got rid of all # greater than 5 and made them 0, so if a number mod 6 equals itself, it was really in the array
            # after getting mod, increase that index by n, so that way as we iterate and come across a # that we added to but havent modded yet, the mod will still remain the same.  so we can increment the corresponding ind
            nums[nums[i]%n]+=n #add n to it so when we check for num//n==0 , it wont trigger the return becuz its been incremented to at least n , n/n=1
        for i in range(1,len(nums)): #0 is not a positive integer so dont bother
            if nums[i]//n==0: #if nums[i]//n ==0 , that means nothin was added to that index from previous loop, and we already threw out all #s greater or equal to n, so if a num//n is 0 that means its untouched
                return i
        return n #edge case where arr=[1,2,3,4,5] so just gotta return the next num which is len([1,2,3,4,5,6]) = 6


def comma(x):
    # if not x:
    #     return(None)
    word=''

    x=str(x)[::-1]

    for i,n in enumerate(x):
        if i!=0 and i%3==0:
            word+=','
        word+=n
    return(word[::-1])

a=comma(23213132)
print(a)
#O(n) linear time , one pass
#O(1) constant, only pointer and incrementing volume
#Trapping Rain Water

if len(height) < 3:
    return (0)

vol = 0
left = 0
right = len(height) - 1

l_max, r_max = height[left], height[right] #set base max as the end bars

while left < right:

    l_max = max(height[left], l_max)
    r_max = max(height[right], r_max)

    if l_max <= r_max: #if left max is lower than the right max,then leftbar[i] is bounded by the L_max becuz the right bound will always be the r_max at the least so ,
        # but if we run into left bar thats higher, than set that as new max, so any followin bars will be bounded by that new max, r_max still gon be the same
        vol += l_max - height[left]
        left += 1
    else:
        vol += r_max - height[right]
        right -= 1

return (vol)

#MEETING ROOMS II
#O(nlogn) time becus of sort , but also n*2logn for extract min or insert worst case
#worst case is that we dont need to allocate any extra room except 1, then have to keep extracting min N times , we pushing N times regardless
#O(n) to create min heap
if not intervals:
    return (0)
intervals.sort(key=lambda x: x[0])  # n log n

freerooms = []
#initialite heap
heapq.heappush(freerooms, intervals[0][1])
#if any room is available, itll be at the top of heap
for i in intervals[1:]:

    if i[0] >= freerooms[0]:
        heapq.heappop(freerooms)  # log N to extract min

    heapq.heappush(freerooms, i[1])  # log N to insert

return (len(freerooms))

#CANDY CRUSH
#O(N*M)**2 time because each scan takes O(3*N*M) cus of the 3 cells we scan for each cell,  in worst case scenario, we only get rid of 3 cells for each iteration,
#which results in worse case # of function calls it takes to clear the board is (N*M)/3 calls since N*M is total # of cells and we keep removing 3 at a time.
#so O(3*N*M) * O((N*M)/3) is O((N*M)**2)

#O(1) Constant space since we just modifying in place

def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
    row = len(board)
    col = len(board[0])
    iterate_again = False

    for i in range(row): #scanning for 3 horizontal matches in a row
        for j in range(col - 2):
            if abs(board[i][j]) == abs(board[i][j + 1]) == abs(board[i][j + 2]) != 0: #cant equal to 0 becuz thats empty cell
                board[i][j] = board[i][j + 1] = board[i][j + 2] = -abs(board[i][j]) #if 3 in a row found, then set value to be its negative form becus we still want to use the #s to find vertical matches
                iterate_again = True #going to have to run function call on board again since we found matches in current board

    for i in range(row - 2): #scanning for 3 vertical matches
        for j in range(col):
            if abs(board[i][j]) == abs(board[i + 1][j]) == abs(board[i + 2][j]) != 0: #have to use abs because the cell may be negative from when we did the horizontal scan
                board[i][j] = board[i + 1][j] = board[i + 2][j] = -abs(board[i][j]) #set its value to be its negative form
                iterate_again = True

    for i in range(col): #drop candies by column startin with first col
        replace_marker = row - 1 #this will mark the next avaiable row that an existing candy in the column will drop down to

        for j in range(row - 1, -1, -1): #start at the last row and fill our way up since gravity makes candy drop
            if board[j][i] > 0: #if the cell has a candy
                board[replace_marker][i] = board[j][i] # assign it to marker of where the next existing candy should drop to
                replace_marker -= 1 #increment the marker upwards to mark next available spot for candy to drop to

        for k in range(replace_marker, -1, -1): #after we moved all the existing candies down, replace marker will be at row where itself and everything above should be empty cells
            board[k][i] = 0 #make empty cell

    return (self.candyCrush(board) if iterate_again else board) #if no matches were found/ drops were made, just reteurn board as it is, or else scan for crushdrops in newly formatted board.


#NUMBERS OF SHIPS IN A RECTANGLE
#O(10*logM*N)
#O(1) in space)
P = topRight
Q = bottomLeft
res = 0
if P.x >= Q.x and P.y >= Q.y and sea.hasShips(P, Q):
    if P.x == Q.x and P.y == Q.y: return 1
    mx, my = (P.x + Q.x) // 2, (P.y + Q.y) // 2
    res += self.countShips(sea, P, Point(mx + 1, my + 1))
    res += self.countShips(sea, Point(mx, P.y), Point(Q.x, my + 1))
    res += self.countShips(sea, Point(mx, my), Q)
    res += self.countShips(sea, Point(P.x, my), Point(mx + 1, Q.y))
return res

#MISSING ELEMENT IN SORTED ARRAY
#O(log n) binary search time
#O(1) constant space
if not nums or not k:
    return (None)

numsrange = nums[-1] - nums[0] + 1  # expected amount of numbers from min num to max num, # can only do this in constant time because of sorted trait
num_missing = numsrange - len(nums)  # amount of #s missing

#have to check if missing number is within range of num array before we can do binary saerch
if k > num_missing:  # if K is greater than the amount of missing numbers in arr, just add on remaining K to last element after deducting the amount of missing # s
    surplus = (k - num_missing)
    return (nums[-1] + surplus)

left = 0
right = len(nums) - 1

while left + 1 < right:  # have to close the gap as much as possible, so once indexes r sidebyside break loop
    mid = left + (right - left) // 2
    missing = nums[mid] - nums[left] - (
                mid - left)  # of missing nums between the two indexes, #expected nums - actual #s present
    if k > missing:  # if k is greater than the # of missing nums between left and mid index, need to search right portion
        left = mid
        k -= missing  # adjust K because we will lose out on the previous misssing #s due to a new starting pt
    else:
        right = mid

return (nums[left] + k)  # eventually gon have two indexes where the missing # lies in between  them,


#BLOOMBERG INTERVIEW QUESTION
def test(board):
    row=len(board)
    col=len(board[0])

    def helper(board,i,j):
        total=0
        total+=(traverse(board,(j+1,col,1),i,True)) #right
        total+=(traverse(board,(j-1,-1,-1),i,True))# left
        total+=(traverse(board,(i-1,-1,-1),j,False))#up
        total+=(traverse(board,(i+1,row,1),j,False)) #bot
        return(total)

    def traverse(board, rng, const, isHoriz):
        i, j, k = rng
        if i < 0  or (i > len(board[0]) - 1 and isHoriz) or (i > len(board) - 1 and not isHoriz):
            return (0)

        spots = 0
        for ind in range(i, j, k):
            val = board[const][ind] if isHoriz else board[ind][const]
            if val == 'X': break
            spots += 1

        return (spots)
    
    mines=[]
    for i in range(row):
        for j in range(col):
            if board[i][j]!='X':
                board[i][j]=helper(board, i, j)
            else:
                mines.append([i,j])
    
    for mine in mines:
        board[mine[0]][mine[1]]=0

    return(board)


graph=[[1, 3, 4,'X'],
       [3 ,4, 7, 2]
      ,[1, 7,'X',3],
       [3,'X',3, 3]]
test(graph)

var = [[5, 4, 3, 0]
       [6, 5, 4, 5],
       [4, 3, 0, 2],
       [3, 0, 1, 3]]
