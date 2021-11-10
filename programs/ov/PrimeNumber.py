# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:29:33 2021

@author: Karthikeyan
"""

num=int(input("Enter a number: "))
flag=0
if(num>1):
    flag=1
for i in range (2,(num-1)):
    if((num%i)==0):
        flag=0
        break
if(num==2):
  flag=1
if(flag==1):
    print(num, "is a prime number")
else:
    print(num,"is not a prime number")
