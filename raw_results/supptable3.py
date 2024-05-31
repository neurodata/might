import numpy as np
import pandas as pd
results='/data/kvhdata1/Projects/MIGHT/Manuscript/Tables'
feature_df=pd.read_csv(
        '/data/kvhdata1/Projects/MIGHT/data/MIGHT.FeatureSets.txt',
        sep=' ',
        names=['Type','Matrix','Directory','Feature','Cardinality'],
        )

feature_df['Directory']=feature_df['Directory'].replace(np.nan,'')
feature_df['path']=[a+':'+b for a,b in list(zip(feature_df['Type'],feature_df['Directory']))]
features=[str(a).strip()+'/'+str(b).strip() for a,b in list(zip(feature_df['Type'],feature_df['Directory']))]
features_df=feature_df.iloc[:15,:]
home='/data/kvhdata1/Projects/MIGHT/results/Cohort1/TestingTreesAndMaxFeatures/results/100000/0.1'
d={}
for feature,name in list(zip(features,feature_df['Feature'])):
    if name in ['DegenerateTetramer','DegenerateHexamer']:continue
    posteriors=pd.read_csv(f'{home}/{feature}/might.posteriors.csv')
    for idx,(sample,posterior) in enumerate(list(zip(posteriors['Sample'],posteriors['Posterior']))):
        try:
            d[sample][name]=posterior
        except:
            d[sample]={'Sample':sample,
                    'Cancer Status':posteriors.loc[idx,'Cancer Status'],
                    'Tumor type':posteriors.loc[idx,'Tumor type'],
                    'Stage':posteriors.loc[idx,'Stage'],
                    }

home='/data/kvhdata1/Projects/MIGHT/results/Cohort1/Ensembles/CustomizedEnsembles/0.1/Wise-1/InsideMonomer'
posteriors=pd.read_csv(f'{home}/might.posteriors.csv')
name='Wise-1+InsideMonomer'
for sample,posterior in list(zip(posteriors['Sample'],posteriors['Posterior'])):
    d[sample][name]=posterior

lst=[]
for sample in d.keys():
    lst.append(d[sample])

df=pd.DataFrame(lst)
df.to_csv(f'{results}/supptable3.csv',index=False)
