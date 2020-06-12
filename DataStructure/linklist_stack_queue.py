#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:33:54 2020

@author: legendary_yin
"""

#linklist  reorder, loop...

class linknode:
    def __init__(self,data):
        self.next = None
        self.value = data
        




class linklist:
    def __init__(self, valuelist):
        self.head = linknode(None)
        currentnode = self.head
        for i in range(len(valuelist)):
            currentnode.next = linknode(valuelist[i])
            currentnode = currentnode.next
            
            
    def reverselinklist(self):
        
        currentnode = self.head.next.next
        prevnode = self.head.next
        next = currentnode.next
        prevnode.next = None
        
        while next is not None:
            currentnode.next = prevnode
            prevnode = currentnode
            currentnode = next
            next = next.next
        
        currentnode.next = prevnode
        self.head.next = currentnode
        return self.head
        
        
    @staticmethod      
    def printlinklist(h):
        currentnode = h.next
        while currentnode is not None:
            print(currentnode.value)
            currentnode = currentnode.next   
        
            
linklist1 = linklist([1,2,3,4,5])
linklist.printlinklist(linklist1.head)

linklist2 = linklist1.reverselinklist()

linklist.printlinklist(linklist2)






#####
# Stack
class stack:
    def __init__(self):
        self.ary = []
        self.top = -1
        self.botton = 0
        self.length = 100
        
    def addelement(self,element):
        if self.top + 1 < self.length:
            self.ary.append(element)
            self.top = self.top + 1
            
        else:
            print('stack full')
    
    
    def popelement(self):
        if self.top == -1:
            print('stack empty')
        else:
            temp = self.ary[-1]
            self.ary = self.ary[:-1]
            self.top = self.top - 1
            return temp
            
data = [1,2,3,4,5,6]
s1 = stack()
for i in data:
    s1.addelement(i)
    
s1.ary
s1.popelement()
s1.ary


paran = dict({')':'(', ']':'[', '}':'{'})
tryeg1 = '{()[]}'
tryeg2 = '{([)]}'

s = stack()
for i in tryeg2:
    if i in ('(','[','{'):
        s.addelement(i)
    else:
        temp = s.popelement()
        if paran[i] != temp:
            print(False)
        
if s.top == -1:
    print(True)
            
    
    
#####
# queue
class queue:
    def __init__(self,length1):
        self.ary = [None] * length1
        self.front = 0
        self.rear = 0
        self.length = length1
        self.fullflag = 0
        
    def addelement(self,element):
        if self.rear == self.front and self.fullflag == 1:
            print('stack full')
            
        elif self.rear == self.length - 1:
            self.ary[self.rear] = element
            self.rear = 0
            
        else:
            self.ary[self.rear] = element
            self.rear = self.rear + 1
            
        if self.rear == self.front:
            self.fullflag = 1
            
            
    
    
    def popelement(self):
        if self.rear == self.front and self.fullflag == 0:
            print('stack empty')
        elif self.front == self.length - 1:
            temp = self.ary[self.front]
            self.ary[self.front] = None
            self.front = 0
        else:
            temp = self.ary[self.front]
            self.ary[self.front] = None
            self.front = self.front + 1
            
        if self.rear == self.front:
            self.fullflag = 0
            
        return temp
            
            
            
            
            
data = [1,2,3,4,5,6]
q1 = queue(6)
for i in data:
    q1.addelement(i)
    
q1.ary
q1.popelement()
q1.popelement()       
q1.addelement(10)        
q1.ary       
q1.addelement(11) 
q1.ary
q1.addelement(13) 
q1.ary    


## LOOP PROBLEM
n = 10
m = 3
k = 2

q2 = queue(n)
for i in range(1,n+1):
    q2.addelement(i)


q2.front = k - 1
q2.rear = k - 1

q2.ary
q2.front
q2.rear
q2.fullflag 


outputcount = 0
while outputcount < n:
    click = 0
    while click < m - 1:
        temp = q2.popelement()
        q2.addelement(temp)
        click = click + 1
        #print(q2.ary)
        #print(q2.front)
        #print(q2.rear)
        
    print(q2.popelement())
    outputcount = outputcount + 1
    #print(q2.ary)
    #print(q2.front)
    #print(q2.rear)
print(q2.popelement())
        
    