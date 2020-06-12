# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:08:44 2018

@author: anel
"""

#basketball spider

import json
import re
import requests
import pandas as pd
import numpy as np

url = 'http://stats.ncaa.org/rankings/MBB/2018/1/145/114'
r=requests.get(url)
m=re.findall('<td class="reclass"  data-order="(.*)"><a href="(.*)" class="skipMask" target="TEAM_WIN">(.*)</a></td>',r.text)
name = [item[0] for item in m] 
web = [item[1] for item in m]
blah = dict(zip(web,name))

data = pd.DataFrame()
total_opponent = []
for i in range(len(blah)):
    url = 'http://stats.ncaa.org' + web[i]
    r=requests.get(url)
    m1=re.findall('<td><a href="(.*)" class="skipMask" target="Rankings">(.*)</a></td>\n    <td align="right">(.*)</td>\n    <td align="right">\n      (.*)\n    </td>',r.text)
    setname = ['Scoring Offense','Scoring Defense','Scoring Margin','Rebound Margin','Assists Per Game','Blocked Shots Per Game','Steals Per Game','Turnover Margin','Assist Turnover Ratio',\
               'Field-Goal Percentage','Field-Goal Percentage Defense','Three-Point Field Goals Per Game','Three-Point Field-Goal Percentage','Free-Throw Percentage',\
               'Won-Lost Percentage']
    nameset = set(setname)
    valuename = set([item[1] for item in m1])
    values = [item[3] for item in m1]
    lost = nameset - valuename 
    
    if lost :
        a = setname.index(list(lost)[0])
        temp = values[:a]
        temp.append('nan')
        temp.extend(values[a:])
        data[web[i]] = temp
    else:
        data[web[i]] = values
    
    m2=re.findall('<a href="(.*)">(.*)</a>\n           </td>\n           <td class="smtext" nowrap>\n                <a href="(.*)" class="skipMask" target="TEAM_WIN">(.*)</a>\n',r.text)
    opponent = [item[0] for item in m2]
    winlose = [item[3][0] for item in m2]
    opponentwinlose = dict(zip(opponent,winlose))
    total_opponent.append(opponentwinlose)
    print(i)

final = pd.DataFrame(columns = ['Result','Scoring Offense','Scoring Defense','Scoring Margin','Rebound Margin','Assists Per Game','Blocked Shots Per Game','Steals Per Game','Turnover Margin','Assist Turnover Ratio',\
               'Field-Goal Percentage','Field-Goal Percentage Defense','Three-Point Field Goals Per Game','Three-Point Field-Goal Percentage','Free-Throw Percentage',\
               'Won-Lost Percentage','Scoring Offense2','Scoring Defense2','Scoring Margin2','Rebound Margin2','Assists Per Game2','Blocked Shots Per Game2','Steals Per Game2','Turnover Margin2','Assist Turnover Ratio2',\
               'Field-Goal Percentage2','Field-Goal Percentage Defense2','Three-Point Field Goals Per Game2','Three-Point Field-Goal Percentage2','Free-Throw Percentage2',\
               'Won-Lost Percentage2'])
for i in range(len(total_opponent)):
    kk = total_opponent[i]  
    
    for j in range(len(kk)):
        temp = []
        tempindex = list(kk.keys())[j]
        finalindex = tempindex.split('/')
        finalindex = '/' + finalindex[1] + '/' + finalindex[2] + '.0' + '/' + finalindex[3]
        if finalindex in list(blah.keys()):
            indexnew = name[i] + 'VS' + blah[finalindex]
            
            temp.append(kk[list(kk.keys())[j]])
            temp.extend(data.iloc[:,i])
            temp.extend(data[finalindex])
            final.loc[indexnew] = temp