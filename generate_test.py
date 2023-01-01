import numpy as np
import pandas as pd
from tqdm import tqdm
import json,os

file_path = 'data/'
test_data =  pd.read_csv(file_path + 'pykt_test.csv')  



#max, len
test_user = np.array(test_data['uid'])
test_dict= {}
for item in np.array(test_data):
    problem = item[1]
    skill = item[2]
    answer = item[3]
    times = item[4]
    repeat = item[5]
    p1 = [int(x) for x in problem.strip().split(',')]
    s1 = [int(x) for x in skill.strip().split(',')]
    a1 = [int(x) for x in answer.strip().split(',')]
    t1 = [int(x) for x in times.strip().split(',')]
    r1 = [int(x) for x in repeat.strip().split(',')]
    if item[1] in test_dict.keys():
        test_dict[item[0]].append([p1, s1, a1, t1, r1])
    else:
        test_dict[item[0]] = []
        test_dict[item[0]].append([p1, s1, a1, t1, r1])

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


user_df_test = []
jiedian = []
for item in tqdm(test_user):
    temp = test_dict[item]
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
    a1 = [x for x in aaa if x != -1]

    his_p = []
    his_kc = []
    his_a = []
    his_ty = []
    his_ana = []
    his_k0 = []
    his_t = []
    his_t1 = []
    his_t2 = []
    his_t3 = []
    his_t4 = []
    his_t5 = []
    his_t6 = []
    his_t7 = []
    his_h = []
    his_h1 = []
    his_h2 = []
    his_h3 = []
    his_h4 = []
    his_h5 = []
    his_h6 = []
    his_h7 = []
    his_al = []
    his_cl = []
    
    for one in range(len(a1)):
        if rrr[one] != 0:
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


        his_a.append(aaa[one])
        his_p.append(ppp[one]+1)
        his_kc.append(sss[one]+1)
        his_ty.append(q2type[ppp[one]]+1)
        his_ana.append(mosts)
        his_k0.append(firstidx[q2first[ppp[one]]])
        his_t.append(tim)
        his_t1.append(time1)
        his_t2.append(time2)
        his_t3.append(time3)
        his_t4.append(time4)
        his_t5.append(time5)
        his_t6.append(time6)
        his_t7.append(time7)
        his_h.append(hope)
        his_h1.append(hope1)
        his_h2.append(hope2)
        his_h3.append(hope3)
        his_h4.append(hope4)
        his_h5.append(hope5)
        his_h6.append(hope6)
        his_h7.append(hope7)

        cl1 = q2content[ppp[one]]
        if cl1 >100:
            cl1 =100
        cl2 = q2analysis[ppp[one]]
        if cl2 >100:
            cl2 =100
        his_cl.append(cl1)
        his_al.append(cl2)

    ccc=0
    pre_p = []
    pre_kc = []
    pre_a = []
    pre_ty = []
    pre_ana = []
    pre_k0 = []
    pre_t = []
    pre_t1 = []
    pre_t2 = []
    pre_t3 = []
    pre_t4 = []
    pre_t5 = []
    pre_t6 = []
    pre_t7 = []
    pre_h = []
    pre_h1 = []
    pre_h2 = []
    pre_h3 = []
    pre_h4 = []
    pre_h5 = []
    pre_h6 = []
    pre_h7 = []
    pre_al = []
    pre_cl = []

    for one in range(len(a1), len(aaa)):
        if rrr[one] != 0:
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

            

        pre_a.append(aaa[one])
        pre_p.append(ppp[one]+1)
        pre_kc.append(sss[one]+1)
        pre_ty.append(q2type[ppp[one]]+1)
        pre_ana.append(mosts)
        pre_k0.append(firstidx[q2first[ppp[one]]])
        pre_t.append(tim)
        pre_t1.append(time1)
        pre_t2.append(time2)
        pre_t3.append(time3)
        pre_t4.append(time4)
        pre_t5.append(time5)
        pre_t6.append(time6)
        pre_t7.append(time7)
        pre_h.append(hope)
        pre_h1.append(hope1)
        pre_h2.append(hope2)
        pre_h3.append(hope3)
        pre_h4.append(hope4)
        pre_h5.append(hope5)
        pre_h6.append(hope6)
        pre_h7.append(hope7)
        cl1 = q2content[ppp[one]]
        if cl1 >100:
            cl1 =100
        cl2 = q2analysis[ppp[one]]
        if cl2 >100:
            cl2 =100
        pre_cl.append(cl1)
        pre_al.append(cl2)
        
        ccc+=1
    jiedian.append(ccc)
    user_df_test.append([his_p, his_kc, his_a, pre_p, pre_kc, pre_a, his_ty, pre_ty, his_ana, pre_ana, his_k0, pre_k0, 
    his_t, pre_t, his_t1, pre_t1, his_t2, pre_t2, his_t3, pre_t3,his_t4, pre_t4,his_t5, pre_t5,his_t6, pre_t6,his_t7, pre_t7,
    his_h, pre_h, his_h1, pre_h1, his_h2, pre_h2, his_h3, pre_h3,his_h4, pre_h4,his_h5, pre_h5,his_h6, pre_h6,his_h7, pre_h7
    ,his_al, pre_al,his_cl, pre_cl])

np.save('data/test.npy', np.array(user_df_test))
np.save('data/jiedian.npy', np.array(jiedian))
print("complete")

