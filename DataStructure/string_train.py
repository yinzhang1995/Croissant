#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:02:28 2020

@author: legendary_yin
"""

# String

# findsubstring
def findsubstring(s, subs):
    if len(subs) > len(s):
        return 0
    for i in range(len(s)):
        for j in range(len(subs)):
            if s[i+j] != subs[j]:
                break
        if j == len(subs) - 1:
            return 1
        
    return 0

findsubstring('goodgoogleiii','google')



# reverse a string
def reversestring(s):
    temp_stack = ''
    final_stack = ''
    
    for i in s[::-1]:
        if i != ' ':
            temp_stack = temp_stack + i
        else:
            for j in temp_stack[::-1]:
                final_stack = final_stack + j
            final_stack = final_stack + ' '
            temp_stack = ''
    final_stack = final_stack + temp_stack[::-1]
    return final_stack
            
reversestring('the sky is blue')



# find the longest common substring
def findlongestsubstring(s1,s2):
    longestsub = ''
    for i in range(len(s1)):
        for j in range(len(s2)):
            templongestsub = ''
            for k in range(j, len(s2)):
                if s1[i + k - j] != s2[k]:
                    break
                else:
                    templongestsub = templongestsub + s2[k]
                    
            if len(templongestsub) > len(longestsub):
                longestsub = templongestsub
        
    return longestsub
    
findlongestsubstring('134562439','123456')



