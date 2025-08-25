import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict

#%% dataweight:mul col name
def ks_calculate(datain, target, score, sort='DES', groupnum=20, tied=1,dataweight=1):

    datain['w'] = datain[dataweight]
    datain['bad'] = datain.loc[datain[target] == 1, 'w']
    datain['good'] = datain.loc[datain[target] == 0, 'w']
    datain[['bad', 'good']] = datain[['bad', 'good']].fillna(0)
    datain['bad_score'] = datain['bad']*datain[score]
    datain['good_score'] = datain['good']*datain[score] 
    datain['score_mul'] = datain['bad_score']+datain['good_score']
    
    #Sort the data
    if sort.upper() == 'DES':
        if pd.__version__ >= '0.17.0':
            datain_sorted = datain.sort_values(score, ascending=False)
        else:
            datain_sorted = datain.sort(score, ascending=False)
    else:
        if pd.__version__ >= '0.17.0':
            datain_sorted = datain.sort_values(score,ascending=True)
        else:
            datain_sorted = datain.sort(score,ascending=True)
    
    #Count total numbers
    nobs_bad = sum(datain.loc[datain[target] == 1, 'w'])
    nobs_good = sum(datain.loc[datain[target] == 0, 'w'])
    nobs_ttl = sum(datain.loc[:, 'w'])
    
    #Divide into groups
    datain_sorted['w_cum'] = np.cumsum(datain_sorted['w'])
    datain_sorted['ctbad'] = np.cumsum(datain_sorted['bad'])
    datain_sorted['ctgood'] = np.cumsum(datain_sorted['good'])
    
    datain_sorted['rank'] = 1
    nobs_grp = float(nobs_ttl) / groupnum
    
    #Whether tied or not
    if tied == 0:
        for i in range(groupnum):
            datain_sorted.loc[datain_sorted['w_cum'] >= i * nobs_grp + 1, 'rank'] = i + 1
    else:
        datain_sorted['diff_score'] = datain_sorted[score].diff()
        datain_sorted['new_index'] = range(1, len(datain_sorted) + 1)
        datain_sorted.set_index('new_index', inplace = True)
        
        w_cum = 0
        index_list = [1]
        for i in range(len(datain_sorted)):
            w_cum += datain_sorted.loc[i + 1, 'w']
            if i + 2 < len(datain_sorted):
                if w_cum >= nobs_grp and datain_sorted.loc[i + 2, 'diff_score'] != 0:
                    #First index of every group
                    index_list.append(i + 2)
                    w_cum -= nobs_grp
            else:
                break
        if groupnum == len(index_list):
            for j in range(groupnum):
                datain_sorted.loc[index_list[j]:, 'rank'] = j + 1
        else:
            print ('There are some problems in grouping!')
    
    #Calculate statistics
    dataout_score = datain_sorted.groupby('rank')[score].aggregate([('minscr', 'min'), ('maxscr', 'max'), ('tscr', 'sum')])
    dataout_score_mul = datain_sorted.groupby('rank')['score_mul'].aggregate([('tscr_mul', 'sum')])
    dataout_tbad = datain_sorted.groupby('rank')['bad'].aggregate([('tbad', 'sum')])
    dataout_tgood = datain_sorted.groupby('rank')['good'].aggregate([('tgood', 'sum')])
    dataout_ctbad = datain_sorted.groupby('rank')['ctbad'].aggregate([('ctbad', 'max')])
    dataout_ctgood = datain_sorted.groupby('rank')['ctgood'].aggregate([('ctgood', 'max')])
    dataout_t = datain_sorted.groupby('rank')['w'].aggregate([('t', 'sum')])
    dataout_ct = datain_sorted.groupby('rank')['w_cum'].aggregate([('ct', 'max')])
    
    #Concatenate all the statistics
    dataout = pd.concat([dataout_score, dataout_score_mul, dataout_tbad, dataout_tgood,dataout_ctbad,
                         dataout_ctgood, dataout_t, dataout_ct],axis = 1)

    #Cummulative percent of total
    dataout['cbadpct'] = dataout['ctbad'] / nobs_bad
    dataout['cgoodpct'] = dataout['ctgood'] / nobs_good
    dataout['ctpct'] = dataout['ct'] / nobs_ttl

    #Percent of total
    dataout['badpct'] = dataout['tbad'] / nobs_bad
    dataout['goodpct'] = dataout['tgood'] / nobs_good
   
    #Interval percent
    dataout['avgbad'] = dataout['tbad'] / dataout['t']
    dataout['avgscr'] = dataout['tscr_mul'] / dataout['t']

    #Cumulative interval percent
    dataout['cavgbad'] = dataout['ctbad'] / dataout['ct']
    
    #Calculate K-S
    dataout['ks_bad_good'] = abs(dataout['cbadpct'] - dataout['cgoodpct'])
    
    #Calculate bumps and decumulative percent
    for i in range(groupnum):

        if i == 0:
            dataout.loc[1, 'bumps'] = 0
        elif dataout.loc[i + 1, 'avgbad'] > dataout.loc[i, 'avgbad']:
            dataout.loc[i + 1, 'bumps'] = dataout.loc[i, 'bumps'] + 1
        else:
            dataout.loc[i + 1, 'bumps'] = dataout.loc[i, 'bumps']

        if i == 0:
            dataout.loc[1, 'dbadpct'] = 1
            dataout.loc[1, 'dgoodpct'] = 1
        else:
            dataout.loc[i + 1, 'dbadpct'] = 1 - dataout.loc[i, 'cbadpct']
            dataout.loc[i + 1, 'dgoodpct'] = 1 - dataout.loc[i, 'cgoodpct']
        
    #Calculate the total line
    dataout.loc['Total', ['minscr', 'maxscr', 'avgscr', 'tscr','tscr_mul', 't', 'tgood', 'tbad', 'goodpct', 
                          'badpct', 'ks_bad_good', 'bumps']] = [
                dataout['minscr'].min(), dataout['maxscr'].max(), dataout['avgscr'].mean(), dataout['tscr'].sum(), dataout['tscr_mul'].sum(), 
                dataout['t'].sum(), dataout['tgood'].sum(), dataout['tbad'].sum(), dataout['goodpct'].sum(), 
                dataout['badpct'].sum(), dataout['ks_bad_good'].max(), dataout['bumps'].max()]

    dataout.loc['Total', 'avgbad'] = dataout.loc['Total', 'tbad'] / dataout.loc['Total', 't']
    
    for i in range(groupnum):
        dataout.loc[i+1, 'Cum Bad Rate Lift'] = dataout.loc[i+1, 'cavgbad']/dataout.loc['Total', 'avgbad']
        dataout.loc[i+1, 'Level Bad Rate Lift'] = dataout.loc[i+1, 'avgbad']/dataout.loc['Total', 'avgbad']
    
    dataout['rank'] = dataout.index
    
    #Rename the dataframe
    dataout.rename(columns={'ks_one_noone':'K-S', 
                            'minscr':'Min Score', 
                            'avgscr':'Mean Score', 
                            'maxscr':'Max Score', 
                            't':'# Total', 
                            'ct':'Cum n', 
                            'ctpct':'Cum % Total',
                            'tdot':'# Ind', 
                            'ctdot':'Cum # Ind', 
                            'ks_bad_good':'K-S',
                            'tbad':'# Bad', 
                            'badpct':'% Total Bad', 
                            'cbadpct':'Cum % Total Bad', 
                            'dbadpct':'Decum % Total Bad', 
                            'tgood':'# Good', 
                            'goodpct':'% Total Good', 
                            'cgoodpct':'Cum % Total Good', 
                            'dgoodpct':'Decum % Total Good', 
                            'ctbad':'Cum # Bad', 
                            'ctgood':'Cum # Good', 
                            'avgbad':'Interval Bad Rate', 
                            'cavgbad':'Cum Bad Rate'}, inplace=True)

    return dataout
#%%
    

def ks_report(datain, target, score, flag=[], outfile='KS_Report.xlsx', sort='DES', groupnum=20, dataweight=1):
    
    dtin=datain.copy(deep=True)
    writer = pd.ExcelWriter(outfile, engine ='xlsxwriter')
    workbook=writer.book
#    format1=workbook.add_format({'align':'center'})
    format2=workbook.add_format({'align':'left'})
    format3=workbook.add_format({'align':'right'})
    format4=workbook.add_format({'align':'right','num_format':'#,##0'})
    format5=workbook.add_format({'align':'right','num_format':'0.00%'})
    format6=workbook.add_format({'align':'right','num_format':'0.000000'})
    format7=workbook.add_format({'align':'right','num_format':'0.0'})
    
    namea="cum top{:.0%} capture bad".format(1/groupnum)
    scorea=score if type(score) is list else [score]
    sortb=[sort]*len(scorea) if type(sort) is str else sort
    sorta={i[0]:i[1] for i in zip(scorea,sortb)}
    flaga=[flag] if type(flag) is str else flag
    flagb={vr:fg for fg in flaga for vr in list(set(datain[fg]))} if len(flaga)>0 else {"":""}

    imp,r1,r2={},0,0
    for sc,vr in product(scorea,flagb):
        if len(vr)==0:
            dataout = ks_calculate(datain=dtin, target=target, score=sc, sort=sorta[sc], groupnum=groupnum,dataweight=dataweight)
        else:
            dataout = ks_calculate(datain=dtin[dtin[flagb[vr]]==vr].reset_index(drop=True), target=target, score=sc, sort=sorta[sc], groupnum=groupnum,dataweight=dataweight)  
        
        dataout['flag'],dataout['score'],imp['flag'],imp['score']=vr,sc,vr,sc
        imp['ks']=dataout['K-S'].max()
        imp[namea]=dataout['Cum % Total Bad'].min()
        dfimp=pd.DataFrame(imp,index=[0],columns=['flag','score','ks',namea])
        
        if r2==0:
            dfimp.to_excel(writer,startrow=r2,index=False,sheet_name='summary')
            r2=r2+2
        else:
            dfimp.to_excel(writer,startrow=r2,index=False,header=0,sheet_name='summary')
            r2=r2+1
        dataout[['rank', 'flag', 'score', 'bumps', '# Total', 'Cum % Total', 'Min Score', 'Mean Score', 
                 'Max Score', '# Good', '# Bad', '% Total Bad', 'Cum % Total Bad', 
                 'Interval Bad Rate', 'K-S', 'Cum Bad Rate', 'Cum Bad Rate Lift', 'Level Bad Rate Lift']].to_excel(writer, sheet_name = 'K-S', startrow=r1, index = 0)
        
        print("{0}-{1} KS: {3:.2%}\n {0}-{1} {2}:{4:.2%}".format(vr,sc,namea,imp['ks'],imp[namea]))
        
        worksheet = writer.sheets['K-S']
        worksheet.conditional_format(r1+1,13,r1+len(dataout)-1,13,{'type':'data_bar','data_bar_2010':True})
        r1=r1+len(dataout)+4
        worksheet = writer.sheets['K-S']
        worksheet.set_column('A:A', 5, format2)
        worksheet.set_column('B:C', 10.5, format2)
        worksheet.set_column('D:D', 10.5, format3)
        worksheet.set_column('E:E', 10.5, format4)
        worksheet.set_column('F:F', 10.5, format5)
        worksheet.set_column('G:G', 10.5, format3)
        if max(dataout['Max Score'])<=1:
            worksheet.set_column('H:H', 10.5, format6)
        else:
            worksheet.set_column('H:H', 10.5, format7)
        worksheet.set_column('I:I', 10.5, format3)
        worksheet.set_column('J:K', 10.5, format4)
        worksheet.set_column('L:P', 10.5, format5)
        worksheet.set_column('Q:R', 10.5, format3)
        
        worksheet=writer.sheets['summary']
        worksheet.set_column('A:B',12,format2)
        worksheet.set_column('C:D',12,format5)
    writer.save()