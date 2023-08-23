import pandas as pd 
from ..Preprocessing.nlp_preprocessing import process_sentence,get_BOMRC
'''
because Pubmed20k had already been processed by jay , so i only rewrite jmir
'''
#%%
'''
this file can create jmir
'''
jmir_df = pd.read_csv("jmir/final_jmir_3inone.csv")
get_abstract_category = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
result = [
    {
        "journalId": i,
        "category": category,
        "ori_sent": jmir_df.loc[i][category],
        "pre_sent": process_sentence(jmir_df.loc[i][category])
    }
    for i in range(len(jmir_df))
    for category in get_abstract_category
]
final_df = pd.DataFrame(result).explode('pre_sent').reset_index(drop=True)
sent_locat_list = []
for i in range(len(jmir_df)):
    count = [j for j in range(1,len(final_df[final_df['journalId']==i])+1)]
    sent_locat_list+=count
    
final_df['sent_locate'] = sent_locat_list
get_location = [get_BOMRC(i) for i in final_df.category.values]
get_location =  pd.DataFrame(get_location,columns=['B','O','M','R','C'])
final_df = pd.concat([final_df,get_location],axis=1)
final_df.to_csv("jmir/new_final_jmir_vertical.csv")
#%%