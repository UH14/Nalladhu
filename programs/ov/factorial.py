# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:27:08 2021

@author: Karthikeyan
"""

num = int(input("Enter a number: "))
factorial = 1
if num<0 :
    print("Sorry, factorial does not exist for negative numbers")
elif(num==0):
    print("The factorial of 0 is 1")
else:
    for i in  range(1,num+1):
        factorial=factorial*i
print("The factorial of",num,"is",factorial)
