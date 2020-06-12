#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:41:23 2020

@author: legendary_yin
"""

#python data manipulation practice

'''
1. Problem:  Member can make purchase via either mobile or desktop platform. 
Using the following data table to determine the total number of member and revenue for mobile-only, 
desktop_only and mobile_desktop.

The input spending table is
member_id    date    channel   spend
1001    1/1/2018    mobile    100
1001    1/1/2018    desktop    100
1002    1/1/2018    mobile    100
1002    1/2/2018    mobile    100
1003    1/1/2018    desktop    100
1003    1/2/2018    desktop    100


The output data is
date    channel    total_spend    total_members
1/1/2018    desktop    100    1
1/1/2018    mobile    100    1
1/1/2018    both    200    1
1/2/2018    desktop    100    1
1/2/2018    mobile    100    1
1/2/2018    both    0    0
'''

'''
sql

select date, count(distinct member_id) as total_members, 
case when mobile_spend > 0 and desktop_spend = 0 then 'mobile' 
when mobile_spend = 0 and desktop_spend > 0 then 'desktop' 
else 'both' end as channel, sum(mobile_spend + desktop_spend) as total_spend
from (
select member_id, date, 
sum(case when channel = 'mobile' then spend else 0 end) as mobile_spend
sum(case when channel = 'desktop' then spend else 0 end) as desktop_spend
from table1
group by member_id,date
) as t1
group by date,
case when mobile_spend > 0 and desktop_spend = 0 then 'mobile' 
when mobile_spend = 0 and desktop_spend > 0 then 'desktop' 
else 'both' end
'''

import pandas as pd
data = {'member_id': [1001,1001,1002,1002,1003,1003], 
        'date': ['1/1/2018','1/1/2018','1/1/2018','1/2/2018','1/1/2018','1/2/2018'],
        'channel': ['mobile','desktop','mobile','mobile','desktop','desktop'],
        'spend': [100,100,100,100,100,100]}
frame = pd.DataFrame(data)

frame['mobile_spend1'] = frame['spend'] * (frame['channel'] ==  'mobile')
frame['desktop_spend1'] = frame['spend'] * (frame['channel'] ==  'desktop')
frame1 = frame.groupby(['member_id','date'],as_index = False).agg({'mobile_spend1':sum,'desktop_spend1':sum})#.to_frame(['mobile_spend','desktop_spend'])
frame1.loc[(frame1['mobile_spend1'] > 0) & (frame1['desktop_spend1'] > 0),'channel'] = 'both'
frame1.loc[(frame1['mobile_spend1'] > 0) & (frame1['desktop_spend1'] == 0),'channel'] = 'moble'
frame1.loc[(frame1['mobile_spend1'] == 0) & (frame1['desktop_spend1'] > 0),'channel'] = 'desktop'


frame2 = frame1.groupby(['date','channel']).agg({'mobile_spend1':sum, 'desktop_spend1':sum})
frame2['spend'] = frame2['mobile_spend1'] + frame2['desktop_spend1']

frame3 = frame1.groupby(['date','channel']).size()
output = pd.concat([frame2, frame3], axis=1)




frame22 = frame1.groupby(['date','channel'],as_index = False).agg({'mobile_spend1':sum, 'desktop_spend1':sum})
frame22['spend'] = frame22['mobile_spend1'] + frame22['desktop_spend1']

frame33 = frame22.groupby(['date','channel']).size().reset_index(name='Size')

output = pd.merge(frame22, frame33, how = 'left',on = ['date','channel'])


a = frame[frame.channel == 'mobile'].spend


'''
Problem: table member_id|company_name|year_start
1): count members who ever moved from Microsoft to Google?
2): count members who directly moved from Microsoft to Google? 
(Microsoft -- Linkedin -- Google doesn't count)
'''

'''
SQL

1) SELECT count(distinct member_id) as num
FROM table t1
left join table t2 on t1,member_id = t2.member_id
where t1.year_start < t2.year_start
and t1.company_name = 'Microsoft'
and t2.company_name = 'Google'

2) 
select * from 
(select member_id,year_start,company_name, rank() over (partition by member_id order by year_start) as rk1 from table) as t1
left join (select member_id,year_start,company_name, rank() over (partition by member_id order by year_start) as rk1 from table) as t2 on t1.member_id = t2.member_id and t1.rk1 = t2.rk1 - 1
where t1.company_name = 'Microsoft'
and t2.company_name = 'Google'



SELECT
COUNT(DISTINCT member_id) AS num_member
FROM
(SELECT
member_id,
company_name,
LEAD(company_name, 1) OVER (PARTITION BY member_id ORDER BY year_start) AS next_company
FROM table
) t
WHERE
Company_name = “Microsoft”
AND next_company = “Google”
'''


table = pd.DataFrame({'member_id':[1,1,2,2,2,3],
                      'company_name':['Microsoft','Google','Microsoft','LinkedIn','Google','ebay'],
                      'year_start':[2001,2002,2009,2010,2011,2012]})

temp = pd.merge(table,table, on = ['member_id'],how='left', suffixes = ('t1_', 't2_'))
len(temp.loc[(temp.company_namet1_ == 'Microsoft') & (temp.company_namet2_ == 'Google') & (temp.year_startt1_ < temp.year_startt2_),'member_id'].unique())


table['next_company'] = table.sort_values(by = ['member_id','year_start']).groupby(['member_id'])['company_name'].shift(-1)


'''
Problem
member_id|email_address, suppose that every member has two email address, get the table in the format of
member_id | email1 | email2
'''

'''
sql

select member_id, max(case when rk = 1 then email_address else NULL end) as email1,
max(case when rk = 2 then email_address else NULL end) as email2 from(
select member_id, email_address, rank() over (partition by member_id order by email_address) as rk) t1
group by member_id
'''

import numpy as np
emailtable = pd.DataFrame({'member_id':[1,1,2,2,3],
                           'email_address':['a','a','c','d','e']})

emailtable['email_rank'] = emailtable.groupby(['member_id'])['email_address'].rank(method = 'dense')

#emailtable.pivot(index='member_id', columns='email_rank', values='email_address')

a = pd.pivot_table(emailtable, values='email_address', index=['member_id'],
                    columns=['email_rank'], aggfunc=np.max, fill_value='').reset_index()

a.rename(columns={1.0:'email1',2.0:'email2'})

a.columns.values

'''
SELECT DISTINCT CALLER
FROM video_calls
where cast(ds as date) >= getdate() - 7
group by caller
order by count(distinct recipient) desc

select count(distinct case when call_id is not null t3.u1 else null end) /  count(distinct t4.user_id)
from dim_all_users t4
left join (
SELECT Caller as u1,recipient as u2,ds,call_id from video_calls t1
union
select receipient as u1, caller as u2,ds,call_id from video_calls t2
) as t3 on t3.u1 = t4.user_id and t3.ds = t4.ds
where country = 'fr'
and cast(ds as date) = getdate() - 1
and dau_flag = 1
'''

video_calls = pd.DataFrame({'caller':['123','032','456'],
                            'recipient':['456','789','032'],
                            'ds':['2020-03-28','2020-03-28','2020-03-28'],
                            'call_id':['4325','9395','0879'],
                            'duration':[864,263,22]})

dim_all_users = pd.DataFrame({'user_id':['123','456','789','032','110'],
                            'age_bucket':['25-34','65+','13-17','45-54','22'],
                            'country':['us','gb','fr','eg','fr'],
                            'primary_os':['android','ios','ios','android','ios'],
                            'dau_flag':[1,1,1,1,1],
                            'ds':['2020-03-28','2020-03-28','2020-03-28','2020-03-28','2020-03-28']})

import datetime

video_calls['ds1'] = pd.to_datetime(video_calls.ds)
dim_all_users['ds1'] = pd.to_datetime(dim_all_users.ds)
v2 = video_calls.groupby(['caller','ds1'])['recipient'].nunique().reset_index()
v2.loc[v2.ds1 >= datetime.date.today() + datetime.timedelta(days = -7),].sort_values(by = ['recipient'],ascending=False).head(10) 


video_calls1 = video_calls.copy()#.iloc[:,[1,0,2,3,4]]
video_calls2 = video_calls1.rename(columns = {'caller':'recipient','recipient':'caller'})
all_call = pd.concat([video_calls,video_calls2], axis = 0)
c = pd.merge(dim_all_users,all_call, left_on=['ds1','user_id'],right_on = ['ds1','caller'], how = 'left')

import numpy as np
len(set(c.loc[(c.ds1 == datetime.date.today() + datetime.timedelta(days = -2)) & (c.country == 'fr') & (c.dau_flag == 1) & (~c.call_id.isnull()),'user_id']))

len(set(c.loc[(c.ds1 == datetime.date.today() + datetime.timedelta(days = -2)) & (c.country == 'fr') & (c.dau_flag == 1),'user_id']))

c.call_id.isnull()



# Flatten a nested list
a = [[1,2],1,[3,[4]],5]

b = [1]
c = [3]
b+c

b.extend(c)   # flattened      [1, 2, 3, 4]
b.append(c)   # not flattend   [1, 2, [3, 4]]

def flattennestedlist(l):
    if type(l) == list and len(l) == 0:
        return []
    #if type(l) == list and len(l) == 1:
        #return l
    if type(l) != list:
        return [l]
    
    first = flattennestedlist(l[0])
    if len(l) > 1:
        remain = flattennestedlist(l[1:])
        return first + remain
    
    return first

a = [[3,[4]]]
a = [[1,2],1,[3,[4]],5]
flattennestedlist(a)
    
def powerproduct(x,n):
    result = 1
    for i in range(n):
        result = result * x
        
    return result

x = 3
n = 4
powerproduct(x,n)



data = pd.DataFrame({'id':[1,2,3,4,5,6,7,8],
                     'visit_date':['2017-01-01','2017-01-02','2017-01-03','2017-01-04',
                                   '2017-01-05','2017-01-06','2017-01-07','2017-01-08'],
                     'people':[10,109,150,99,145,1355,199,188]})
    
    
data1 = data.sort_values(by = ['visit_date'])

newcol = [0]

for i in range(1,len(data1['id'])):
    if data1['people'][i] >= 100 and data1['people'][i - 1] >= 100:
        temp = newcol[-1]
        newcol.append(temp)
    else:
        temp = newcol[-1]
        newcol.append(temp + 1)
        
data1['newcol'] = newcol
c = data1.groupby(['newcol']).size().reset_index()
d = c.rename(columns = {0:'count'})

e = pd.merge(data1, d, on = ['newcol'], how = 'left')
e.loc[e['count'] >= 3,['id','visit_date','people']]



import numpy as np
trips = pd.DataFrame({'Id':np.arange(1,11),
              'Client_Id':[1,2,3,4,1,2,3,2,3,4],
              'Driver_Id':[10,11,12,13,10,11,12,12,10,13],
              'City_Id':[1,1,6,6,1,6,6,12,12,12],
              'Status':(['Completed','cancelled_by_driver','Completed','cancelled_by_client'] + ['Completed'] * 5 + ['cancelled_by_driver']),
              'Request_at':['2013-10-01']*4 + ['2013-10-02']*3 + ['2013-10-03']*3})

users = pd.DataFrame({'Users_Id':[1,2,3,4,10,11,12,13],
              'Banned':['No','Yes'] + ['No'] * 6,
              'Role':['client'] * 4 + ['driver'] * 4})

trips['Request_at'] = pd.to_datetime(trips['Request_at'])
temp1 = pd.merge(trips, users,left_on = ['Client_Id'], right_on = ['Users_Id'],how = 'left')
temp2 = pd.merge(temp1, users,left_on = ['Driver_Id'], right_on = ['Users_Id'],how = 'left',suffixes =('1','2') )

temp3 = temp2.loc[(temp2.Banned1 == 'No') & (temp2.Banned2 == 'No'),].groupby(['Request_at']).size().reset_index()
temp3 = temp3.rename(columns = {0:'totaltrip'})

temp4 = temp2.loc[(temp2.Banned1 == 'No') & (temp2.Banned2 == 'No') & (temp2.Status != 'Completed'),].groupby(['Request_at']).size().reset_index()
temp4 = temp4.rename(columns = {0:'canceltrip'})

temp5 = pd.merge(temp3, temp4, on = ['Request_at'], how = 'left').fillna(0)
temp5['cancellationrate'] = round(temp5.canceltrip / temp5.totaltrip,2)

a = pd.pivot_table(trips, index = ['Request_at'], columns = ['Status'], values = ['Id'], aggfunc='count',fill_value = 0)
a.columns
a.reset_index(inplace = True)
a
pd.melt(a,id_vars = ['Request_at'],value_vars = ['Completed','cancelled_by_client','cancelled_by_driver'])



# Creating the Series 
sr = pd.Series([10, 25, 3, 25, 24, 6]) 
  
# Create the Index 
index_ = ['Coca Cola', 'Sprite', 'Coke', 'Fanta', 'Dew', 'ThumbsUp'] 
  
# set the index 
sr.index = index_ 
sr.keys() 
sr.values

sr['Coca Cola']

sr['new'] = 14


sr_dict = dict(sr)
sr_dict.keys()
sr_dict.values

sr_dict['new']

index_ = pd.date_range('2010-10-09', periods = 11, freq ='M') 

if 'old' in sr_dict.keys():
    a = 0
else:
    a = 1
    
    
import pandas as pd
employee = pd.DataFrame({'Id': [1,2,3,4,5,6,7],
                         'Name':['a','b','c','d','e','f','g'],
                         'Salary':[85000,80000,60000,90000,69000,85000,70000],
                         'DepartmentID':[1,2,2,1,1,1,1]})
    
department = pd.DataFrame({'Id':[1,2],
                           'Name':['IT','Sales']}) 
    
df = pd.merge(employee, department, left_on = ['DepartmentID'], right_on = ['Id'], how = 'left')
df.sort_values(['DepartmentID','Salary'],ascending = False).groupby(['DepartmentID']).head(3)

df['grouprank'] = df.groupby(['DepartmentID'])['Salary'].rank(method = 'dense', ascending = False)
result1 = df.loc[df.grouprank <= 3,['Name_y','Name_x','Salary']]
result1.rename(columns = {'Name_y':'Department','Name_x':'Employee'})
 