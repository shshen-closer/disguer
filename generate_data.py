import numpy as np
import pandas as pd
from tqdm import tqdm
import json,os
import sys

file_path = 'data/'
all_data =  pd.read_csv(file_path + 'train_valid_sequences.csv')  

user = np.array(all_data['uid'])
user = list(set(user))

data_dict= {}

for item in np.array(all_data):
    problem = item[2]
    skill = item[3]
    answer = item[4]
    times = item[5]
    repeat = item[7]
    p1 = [int(x) for x in problem.strip().split(',')]
    s1 = [int(x) for x in skill.strip().split(',')]
    a1 = [int(x) for x in answer.strip().split(',')]
    t1 = [int(x) for x in times.strip().split(',')]
    r1 = [int(x) for x in repeat.strip().split(',')]
    if item[1] in data_dict.keys():
        data_dict[item[1]].append([p1, s1, a1, t1, r1])
    else:
        data_dict[item[1]] = []
        data_dict[item[1]].append([p1, s1, a1, t1, r1])


with open(file_path + 'questions.json') as fi:
    for line in fi:
        qinfo = json.loads(line)
with open(file_path + 'keyid2idx.json') as fi:
    for line in fi:
        keyid2idx = json.loads(line)
q2id = keyid2idx['questions']
k2id = keyid2idx['concepts']
q2type = {}
for item in qinfo.keys():
    q2type[q2id[item]] = qinfo[item]['type']

first_c = []
q2first = {}
for item in qinfo.keys():
    q2first[q2id[item]] = int(qinfo[item]['concept_routes'][0].split('----')[0])
    first_c.append(int(qinfo[item]['concept_routes'][0].split('----')[0]))
first_c = list(set(first_c))
firstidx = {}
idxs = 1
for sss in first_c:
    firstidx[sss] = idxs
    idxs+=1


q2analysis = {}
a2fisrt = {}
ana_len = []
for item in qinfo.keys():
    temp = qinfo[item]['analysis']
    ana_len.append(len(temp))
    a2fisrt[q2id[item]] = len(temp)
    while 1902 in temp:
        temp.remove(1902)
    while 832 in temp:
        temp.remove(832)
    while 2718 in temp:
        temp.remove(2718)
    q2analysis[q2id[item]] = len(temp)
ana_lens = np.mean(ana_len)/3

q2content = {}
for item in qinfo.keys():
    temp = qinfo[item]['content']
    while 1902 in temp:
        temp.remove(1902)
    while 832 in temp:
        temp.remove(832)
    while 2718 in temp:
        temp.remove(2718)
    q2content[q2id[item]] = len(temp)

all_data = []




for item in tqdm(user):
    temp = data_dict[item]
    ppp = []
    sss = []
    aaa = []
    ttt = []
    rrr = []
    for iii in temp:
        ppp += iii[0]
        sss += iii[1]
        aaa += iii[2]
        ttt += iii[3]
        rrr += iii[4]
    aaa = [x for x in aaa if x != -1]
    ppp = ppp[:len(aaa)]
    sss = sss[:len(aaa)]
    ttt = ttt[:len(aaa)]
    rrr = rrr[:len(aaa)]

    all_p = []
    all_kc = []
    all_a = []
    all_ty = []
    all_ana = []
    all_time = []
    all_time1 = []
    all_time2 = []
    all_time3 = []
    all_time4 = []
    all_time5 = []
    all_time6 = []
    all_time7 = []
    all_hope = []
    all_hope1 = []
    all_hope2 = []
    all_hope3 = []
    all_hope4 = []
    all_hope5 = []
    all_hope6 = []
    all_hope7 = []
    all_k0 = []
    all_cl = []
    all_al = []

    for one in range(len(aaa)):
        if rrr[one] !=0:
            continue

        if  a2fisrt[ppp[one]] < ana_lens:
            mosts = 1
        elif a2fisrt[ppp[one]] < 2*ana_lens:
            mosts = 2
        else:
            mosts = 3


        if one > 0:
            tim = int((ttt[one]-ttt[one-1])/10000) + 1
        else:
            tim = 102
        if tim > 100:
            tim = 101
        if one > 1:
            time1 = int((ttt[one]-ttt[one-2])/20000) + 1
        else:
            time1 = 102
        if time1 > 100:
            time1 = 101
        if one > 2:
            time2 = int((ttt[one]-ttt[one-3])/30000) + 1
        else:
            time2 = 102
        if time2 > 100:
            time2 = 101
        if one > 3:
            time3 = int((ttt[one]-ttt[one-4])/40000) + 1
        else:
            time3 = 102
        if time3 > 100:
            time3 = 101
        if one > 4:
            time4 = int((ttt[one]-ttt[one-5])/50000) + 1
        else:
            time4 = 102
        if time4 > 100:
            time4 = 101
        if one > 5:
            time5 = int((ttt[one]-ttt[one-6])/60000) + 1
        else:
            time5 = 102
        if time5 > 100:
            time5 = 101
        if one > 6:
            time6 = int((ttt[one]-ttt[one-7])/70000) + 1
        else:
            time6 = 102
        if time6 > 100:
            time6 = 101
        if one > 7:
            time7 = int((ttt[one]-ttt[one-8])/80000) + 1
        else:
            time7 = 102
        if time7 > 100:
            time7 = 101
        
        if one < len(aaa)-1:
            hope = int((ttt[one+1]-ttt[one])/10000) + 1
        else:
            hope = 102
        if hope > 100:
            hope = 101
        if one < len(aaa)-2:
            hope1 = int((ttt[one+2]-ttt[one])/20000) + 1
        else:
            hope1 = 102
        if hope1 > 100:
            hope1 = 101
        if one < len(aaa)-3:
            hope2 = int((ttt[one+3]-ttt[one])/30000) + 1
        else:
            hope2 = 102
        if hope2 > 100:
            hope2 = 101
        if one < len(aaa)-4:
            hope3 = int((ttt[one+4]-ttt[one])/40000) + 1
        else:
            hope3 = 102
        if hope3 > 100:
            hope3 = 101
        if one < len(aaa)-5:
            hope4 = int((ttt[one+5]-ttt[one])/50000) + 1
        else:
            hope4 = 102
        if hope4 > 100:
            hope4 = 101
        if one < len(aaa)-6:
            hope5 = int((ttt[one+6]-ttt[one])/60000) + 1
        else:
            hope5 = 102
        if hope5 > 100:
            hope5 = 101
        if one < len(aaa)-7:
            hope6 = int((ttt[one+7]-ttt[one])/70000) + 1
        else:
            hope6 = 102
        if hope6 > 100:
            hope6 = 101
        if one < len(aaa)-8:
            hope7 = int((ttt[one+8]-ttt[one])/80000) + 1
        else:
            hope7 = 102
        if hope7 > 100:
            hope7 = 101
    
        all_a.append(aaa[one])
        all_p.append(ppp[one]+1)
        all_kc.append(sss[one]+1)
        all_ty.append(q2type[ppp[one]]+1)
        all_ana.append(mosts)
        all_k0.append(firstidx[q2first[ppp[one]]])
        all_time.append(tim)
        all_time1.append(time1)
        all_time2.append(time2)
        all_time3.append(time3)
        all_time4.append(time4)
        all_time5.append(time5)
        all_time6.append(time6)
        all_time7.append(time7)
        all_hope.append(hope)
        all_hope1.append(hope1)
        all_hope2.append(hope2)
        all_hope3.append(hope3)
        all_hope4.append(hope4)
        all_hope5.append(hope5)
        all_hope6.append(hope6)
        all_hope7.append(hope7)
        cl1 = q2content[ppp[one]]
        if cl1 >100:
            cl1 =100
        cl2 = q2analysis[ppp[one]]
        if cl2 >100:
            cl2 =100
        all_cl.append(cl1)
        all_al.append(cl2)
        




    
    pre_p = []
    pre_kc = []
    pre_a = []
    pre_ty = []
    pre_ana = []
    pre_k0 = []
    pre_time = []
    pre_time1 = []
    pre_time2 = []
    pre_time3 = []
    pre_time4 = []
    pre_time5 = []
    pre_time6 = []
    pre_time7 = []
    pre_hope = []
    pre_hope1 = []
    pre_hope2 = []
    pre_hope3 = []
    pre_hope4 = []
    pre_hope5 = []
    pre_hope6 = []
    pre_hope7 = []
    pre_al = []
    pre_cl = []
   
    for one in range(len(all_a)):
        pre_p.append(all_p[one])
        pre_kc.append(all_kc[one])
        pre_a.append(all_a[one])
        pre_ty.append(all_ty[one])
        pre_ana.append(all_ana[one])
        pre_time.append(all_time[one])
        pre_time1.append(all_time1[one])
        pre_time2.append(all_time2[one])
        pre_time3.append(all_time3[one])
        pre_time4.append(all_time4[one])
        pre_time5.append(all_time5[one])
        pre_time6.append(all_time6[one])
        pre_time7.append(all_time7[one])
        pre_hope.append(all_hope[one])
        pre_hope1.append(all_hope1[one])
        pre_hope2.append(all_hope2[one])
        pre_hope3.append(all_hope3[one])
        pre_hope4.append(all_hope4[one])
        pre_hope5.append(all_hope5[one])
        pre_hope6.append(all_hope6[one])
        pre_hope7.append(all_hope7[one])
        pre_k0.append(all_k0[one])
        pre_al.append(all_al[one])
        pre_cl.append(all_cl[one])

    all_data.append([pre_p, pre_kc, pre_a, pre_ty, pre_ana, pre_k0, pre_time, pre_time1, pre_time2, pre_time3, pre_time4, 
    pre_time5, pre_time6, pre_time7, pre_hope, pre_hope1, pre_hope2, pre_hope3, pre_hope4, pre_hope5, pre_hope6, pre_hope7, pre_al, pre_cl])


np.save('data/all_data.npy', np.array(all_data))
print("complete")

