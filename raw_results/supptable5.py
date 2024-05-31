import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import entropy, ortho_group
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from numpy.typing import ArrayLike
import pandas as pd

results='/data/kvhdata1/Projects/MIGHT/Manuscript/Tables'

def _mutual_information(y_true: ArrayLike, y_pred_proba: ArrayLike) -> float:
    """Compute estimate of mutual information for supervised classification setting.
    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        The true labels.
    y_pred_proba : ArrayLike of shape (n_samples, n_outputs)
        Posterior probabilities.
    Returns
    -------
    float :
        The estimated MI.
    """
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    # entropy averaged over n_samples
    H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def get_performance(features,names,home,card):
    lst=[]
    missing=[]
    posteriors=None
    for idx,feature in enumerate(features):
        print(feature)
        try:
            #mi=pd.read_csv(f'{home}/{feature}/might.metrics.csv')['MI'].tolist()[0]
            n_estimators=pd.read_csv(f'{home}/{feature}/might.metrics.csv')['n_estimators'].tolist()[0]
            df=pd.read_csv(f'{home}/{feature}/might.posteriors.csv')
        except:
            print(f'Could not find file for {feature}')
            missing.append(feature)
            continue
        mi = _mutual_information(df['Cancer Status'].to_numpy(),np.array([[1-x,x] for x in df['Posterior'].tolist()]))
        print(df)
        print(names[idx])
        type=names[idx].split('/')[0]
        v1=names[idx].split('/')[1]
        v2=names[idx].split('/')[2]
        v3=names[idx].split('/')[3]
        print(feature,v1,v2)
        if posteriors is None:
            posteriors=df.iloc[:,:-1]
        print(card[idx])
        v1c=card[idx][0]
        v2c=card[idx][1]
        v3c=card[idx][2]
        if v3!='':
            posteriors[v1+'+'+v2+'+'+v3]=df['Posterior']
        if v2!='':
            posteriors[v1+'+'+v2]=df['Posterior']
        else:
            posteriors[v1]=df['Posterior']
        auc = roc_auc_score(df['Cancer Status'], df['Posterior'])
        p_auc_90 = roc_auc_score(df['Cancer Status'], df['Posterior'],max_fpr=0.10)
        p_auc_95 = roc_auc_score(df['Cancer Status'], df['Posterior'],max_fpr=0.05)
        p_auc_98 = roc_auc_score(df['Cancer Status'], df['Posterior'],max_fpr=0.02)
        fpr, tpr, thresholds = roc_curve(df['Cancer Status'], df['Posterior'])
        fpr_tpr = list(zip(fpr, tpr))
        s90 = max([tpr for (fpr,tpr) in fpr_tpr if fpr <= 0.1])
        s95 = max([tpr for (fpr,tpr) in fpr_tpr if fpr <= 0.05])
        s98 = max([tpr for (fpr,tpr) in fpr_tpr if fpr <= 0.02])
        lst.append({
                    'Variable Type':type,
                    'Variable Set 1':v1,
                    'Variable Set 2':v2,
                    'Variable Set 3':v3,
                    '# of features in Variable Set 1':v1c,
                    '# of features in Variable Set 2':v2c,
                    '# of features in Variable Set 3':v3c,
                    'Total # of variables':(v1c+v2c+v3c),
                    'MI':mi,
                    'AUROC':auc,
                    'S@98':s98,
                    })
    print(pd.DataFrame(lst))
    return lst, posteriors, missing

###################
### Single-View ###
###################
feature_df=pd.read_csv(
        '/data/kvhdata1/Projects/MIGHT/data/MIGHT.FeatureSets.txt',
        sep=' ',
        names=['Type','Matrix','Directory','Feature','Cardinality'],
        )
feature_df['Directory']=feature_df['Directory'].replace(np.nan,'')
feature_df=feature_df[feature_df['Type']!='ichorCNA']
feature_df=feature_df[~feature_df['Feature'].isin(['DegenerateTetramer','DegenerateHexamer'])]
feature_df['path']=[a+':'+b for a,b in list(zip(feature_df['Type'],feature_df['Directory']))]
cardinality_df={a+':'+b:c for a,b,c in list(zip(
                                    feature_df['Type'],
                                    feature_df['Directory'],
                                    feature_df['Cardinality']
                                    ))
                    }
features=[str(a).strip()+'/'+str(b).strip() for a,b in list(zip(feature_df['Type'],feature_df['Directory']))]
print(features)
names=feature_df['Feature'].tolist()
names=[a+'/'+b+'//' for a,b in list(zip(feature_df['Type'],feature_df['Feature']))]
card = [[c,0,0] for c in feature_df['Cardinality'].tolist()]
lst=[]
missing=[]
for tissue in ['Breast','Pancreas']:
    home=f'/data/kvhdata1/Projects/MIGHT/results/{tissue}StageII/TestingTreesAndMaxFeatures/results/100000/0.1'
    perf,post,missing=get_performance(features,names,home,card)
    perf=pd.DataFrame(perf)
    perf['Tissue']=tissue
    lst.append(perf)

singleview=pd.concat(lst)
singleview.insert(0,'Type of MIGHT','MIGHT')
singleview['Variable Type']=singleview['Variable Type'].replace('LociFraction','Genomic Annotation')
singleview['Variable Type']=singleview['Variable Type'].replace('LengthAnalysis','Fragment Length')
singleview['Variable Type']=singleview['Variable Type'].replace('MotifAnalysis','Fragment Motif')
singleview['Variable Type']=singleview['Variable Type'].replace('WiseCondorX','Aneuploidy')
singleview['Variable Type']=singleview['Variable Type'].replace('Delfi','Fragment Length')

################
### Two-View ###
################
comight=pd.read_csv('/data/kvhdata1/Projects/MIGHT/data/CoMIGHT.FeatureSets.txt',
                    sep=' ',
                    header=None)
comight=comight[comight[0]!='ichorCNA']
features=comight[2].tolist()
comight_features=[]
names=[]
for feature in features:
    for feature2 in features:
        if feature==feature2:continue
        if feature.startswith('Outside') and feature2.startswith('Outside'):continue
        if feature.startswith('Inside') and feature2.startswith('Inside'):continue
        if feature.startswith('Inside') and feature2.startswith('Congruent'):continue
        if feature.startswith('Congruent') and feature2.startswith('Inside'):continue
        if feature.startswith('Outside') and feature2.startswith('Congruent'):continue
        if feature.startswith('Congruent') and feature2.startswith('Outside'):continue
        if feature.startswith('Congruent') and feature2.startswith('Congruent'):continue
        if feature.startswith('InsideTrimer') and feature2.startswith('InsideTrimer'):continue
        if feature.startswith('Wise') and feature2.startswith('Wise'):continue
        comight_features.append(feature+'/'+feature2)
        names.append('TwoView'+'/'+feature+'/'+feature2+'/')

print(comight_features)
card = [[cardinality_df[a],cardinality_df[b],0] for a,b in list(product(feature_df['path'].tolist(),repeat=2))]
lst=[]
posteriors=[]
for tissue in ['Breast','Pancreas']:
    home=f'/data/kvhdata1/Projects/MIGHT/results/{tissue}StageII/Ensembles/CustomizedEnsembles/0.1'
    perf,post,missing=get_performance(comight_features,names,home,card)
    perf=pd.DataFrame(perf)
    perf['Tissue']=tissue
    lst.append(perf)

twoview=pd.concat(lst)
twoview.insert(0,'Type of MIGHT','CoMIGHT')

##################
### Three View ###
##################
lst=[]
for tissue in ['Breast','Pancreas']:
    dir=f'/data/kvhdata1/Projects/MIGHT/results/{tissue}StageII/Ensembles/CustomizedEnsembles/0.1'
    top=pd.read_csv(f'{dir}/topperformingtwoview.csv',sep='\t',header=None).iloc[:1,:]
    threeview_features=[]
    for feature1,feature2 in list(zip(top[0],top[1])):
        threeview_features.extend(['/'.join([feature1,feature2,feature3]) for feature3 in features])
    threeview_names=[]
    for feature1,feature2 in list(zip(top[0],top[1])):
        threeview_names.extend(['ThreeView'+'/'+'/'.join([feature1,feature2,feature3]) for feature3 in features])
    threeview_card=[]
    for feature1,feature2 in list(zip(top[0],top[1])):
        c1=comight[comight[2]==feature1][0].tolist()[0]+':'+comight[comight[2]==feature1][3].tolist()[0]
        c2=comight[comight[2]==feature2][0].tolist()[0]+':'+comight[comight[2]==feature2][3].tolist()[0]
        threeview_card.extend([[cardinality_df[c1],
                                cardinality_df[c2],
                                cardinality_df[c3]]for c3 in feature_df['path'].tolist()])
    home=f'/data/kvhdata1/Projects/MIGHT/results/{tissue}StageII/Ensembles/CustomizedEnsembles/0.1'
    perf,post,missing=get_performance(threeview_features,threeview_names,home,threeview_card)
    perf=pd.DataFrame(perf)
    perf['Tissue']=tissue
    print(perf)
    lst.append(perf)

threeview=pd.concat(lst)
threeview.insert(0,'Type of MIGHT','CoMIGHT')

df=pd.concat([singleview,twoview,threeview])

df.index=range(len(df))

for i in ['Variable Set 1','Variable Set 2','Variable Set 3']:
    df[i]=df[i].replace('CompartmentFraction','GM12878 A/B Genome Compartments')
    df[i]=df[i].replace('LINEsFractino','LINEsFraction')
    df[i]=df[i].replace('cCRE_Promoters','Promoter cCREs')
    df[i]=df[i].replace('cCRE_DistalEnhancers','DistalEnhancer cCREs')
    df[i]=df[i].replace('cCRE_ProximalEnhancers','ProximalEnhancer cCREs')
    df[i]=df[i].replace('cCRE_DNaseH3K4me3','DNaseH3K4me3 cCREs')
    df[i]=df[i].replace('cCRE_CTCF','CTCF cCREs')
    df[i]=df[i].replace('cCREFraction','Candidate Cis-Regulatory Elements')
    df[i]=df[i].replace('Length1','Length-1')
    df[i]=df[i].replace('Length5','Length-5')
    df[i]=df[i].replace('Length10','Length-10')
    df[i]=df[i].replace('Length15','Length-15')
    df[i]=df[i].replace('Length20','Length-20')
    df[i]=df[i].replace('LengthRatio','Fragment Length Ratio')
    df[i]=df[i].replace('AluS','AluS Elements')
    df[i]=df[i].replace('AluY','AluY Elements')
    df[i]=df[i].replace('AluJ','AluJ Elements')
    df[i]=df[i].replace('AluFraction','Alus')
    df[i]=df[i].replace('InsideTrimer0.01','InsideTrimer 1% GC Bin')
    df[i]=df[i].replace('InsideTrimer0.05','InsideTrimer 5% GC Bin')
    df[i]=df[i].replace('InsideTrimer0.10','InsideTrimer 10% GC Bin')
    df[i]=df[i].replace('InsideTrimer0.15','InsideTrimer 15% GC Bin')


out=[]
for idx in range(len(df)):
    feature=df.loc[idx,'Variable Set 1']
    feature2=df.loc[idx,'Variable Set 2']
    feature3=df.loc[idx,'Variable Set 3']
    if feature==feature2:continue
    if feature.startswith('Outside') and feature2.startswith('Outside'):continue
    if feature.startswith('Inside') and feature2.startswith('Inside'):continue
    if feature.startswith('Inside') and feature2.startswith('Congruent'):continue
    if feature.startswith('Congruent') and feature2.startswith('Inside'):continue
    if feature.startswith('Outside') and feature2.startswith('Congruent'):continue
    if feature.startswith('Congruent') and feature2.startswith('Outside'):continue
    if feature.startswith('Congruent') and feature2.startswith('Congruent'):continue
    if feature.startswith('InsideTrimer') and feature2.startswith('InsideTrimer'):continue
    if feature.startswith('Wise') and feature2.startswith('Wise'):continue
    if feature.startswith('Length-') and feature2.startswith('Length-'):continue
    if feature.startswith('Outside') and feature3.startswith('Outside'):continue
    if feature.startswith('Inside') and feature3.startswith('Inside'):continue
    if feature.startswith('Inside') and feature3.startswith('Congruent'):continue
    if feature.startswith('Congruent') and feature3.startswith('Inside'):continue
    if feature.startswith('Outside') and feature3.startswith('Congruent'):continue
    if feature.startswith('Congruent') and feature3.startswith('Outside'):continue
    if feature.startswith('Congruent') and feature3.startswith('Congruent'):continue
    if feature.startswith('InsideTrimer') and feature3.startswith('InsideTrimer'):continue
    if feature.startswith('Wise') and feature3.startswith('Wise'):continue
    if feature.startswith('Length-') and feature3.startswith('Length-'):continue
    if feature3.startswith('Outside') and feature2.startswith('Outside'):continue
    if feature3.startswith('Inside') and feature2.startswith('Inside'):continue
    if feature3.startswith('Inside') and feature2.startswith('Congruent'):continue
    if feature3.startswith('Congruent') and feature2.startswith('Inside'):continue
    if feature3.startswith('Outside') and feature2.startswith('Congruent'):continue
    if feature3.startswith('Congruent') and feature2.startswith('Outside'):continue
    if feature3.startswith('Congruent') and feature2.startswith('Congruent'):continue
    if feature3.startswith('InsideTrimer') and feature2.startswith('InsideTrimer'):continue
    if feature3.startswith('Wise') and feature2.startswith('Wise'):continue
    if feature3.startswith('Length-') and feature2.startswith('Length-'):continue
    out.append(df.loc[idx,:])

df=pd.DataFrame(out)


descriptions=pd.read_csv('/data/kvhdata1/Projects/MIGHT/data/featuredescriptions.txt',
                        sep='\t',
                        header=None)

descriptions_df={a:b for a,b in list(zip(descriptions[0],descriptions[1]))}
abbrevations_df={a:b for a,b in list(zip(descriptions[0],descriptions[2]))}

d1=df[df['Type of MIGHT']=='MIGHT']
d1['Abbreviation']=d1['Variable Set 1'].apply(lambda x:abbrevations_df[x])
d1['Description']=d1['Variable Set 1'].apply(lambda x:descriptions_df[x])


d2=df[df['Variable Type']=='TwoView']
d2['Abbreviation']=[abbrevations_df[a]+'+'+abbrevations_df[b] for a,b in list(zip(d2['Variable Set 1'],d2['Variable Set 2']))]
d2['Description']=[f'Combination of {a} and {b}' for a,b in list(zip(d2['Variable Set 1'],d2['Variable Set 2']))]

d3=df[df['Variable Type']=='ThreeView']
d3['Abbreviation']=[abbrevations_df[a]+'+'+abbrevations_df[b]+'+'+abbrevations_df[c] for a,b,c in list(zip(d3['Variable Set 1'],d3['Variable Set 2'],d3['Variable Set 3']))]
d3['Description']=[f'Combination of {a} and {b} and {c}' for a,b,c in list(zip(d3['Variable Set 1'],d3['Variable Set 2'],d3['Variable Set 3']))]

out=pd.concat([d1,d2,d3])

out.to_csv(f'{results}/supptable5.csv',index=False)
