import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_palette("bright")
sns.set(font_scale=1.1)
bright_colors = sns.color_palette("bright", 6)
sns.set_style('white')

variable_order=[
            'Fragment Motif',
            'Fragment Length',
            'Breakpoint Analysis',
            'Genomic Annotation',
            'Aneuploidy'
            ]


plots=[]

################################################################################

                            ################
                            ### Figure 3 ###
                            ################

################################################################################

#dir='/data/kvhdata1/Projects/MIGHT/results/ByStage/StageIV'
dir='./'
outdir='./figs'
mv=pd.read_csv(f'{dir}/122723_ConsolidatedMIGHT.multiview.metrics.csv')
pt=pd.read_csv(f'{dir}/122723_ConsolidatedMIGHT.singleview.metrics.perTree.csv')
df=pd.read_csv(f'{dir}/122723_ConsolidatedMIGHT.singleview.metrics.csv')
df['Variable Type']=df['Feature Type'].replace('Chromatin Accessibility','Breakpoint Analysis')
df['Variable Type']=df['Variable Type'].replace('Delfi','Fragment Length')

d={a:b for a,b in list(zip(df['Feature'],df['Variable Type']))}
pt['Variable Type']=pt['Feature'].apply(lambda x:d[x])
d={a:b for a,b in list(zip(df['Feature'],df['# Features']))}
pt['# Features']=pt['Feature'].apply(lambda x:d[x])
df['Feature']=df['Feature'].replace('LengthBin 1','Length-1')
df['Feature']=df['Feature'].replace('LengthBin 5','Length-5')
df['Feature']=df['Feature'].replace('LengthBin 10','Length-10')
df['Feature']=df['Feature'].replace('LengthBin 15','Length-15')
df['Feature']=df['Feature'].replace('LengthBin 20','Length-20')
df['Feature']=df['Feature'].replace('WiseCondorX_1Mb','Wise-1')
df['Feature']=df['Feature'].replace('WiseCondorX_5Mb','Wise-5')
df['Feature']=df['Feature'].replace('WiseCondorX_10Mb','Wise-10')
df['Feature']=df['Feature'].replace('GenomeCompartmentsFraction','A/B Compartments')
df['Feature']=df['Feature'].replace('AlusFraction','Alu Fraction')
df['Feature']=df['Feature'].replace('LINEsFraction','LINE Fraction')
df['Feature']=df['Feature'].replace('FunctionalElementsFraction','cCRE')
df['Feature']=df['Feature'].replace('cCRE_Promoters','Promoters')
df['Feature']=df['Feature'].replace('cCRE_ProximalEnhancers','ProximalEnhancers')
df['Feature']=df['Feature'].replace('cCRE_DistalEnhancers','DistalEnhancers')
df['Feature']=df['Feature'].replace('cCRE_DNaseH3K4me3','DNaseH3K4me3')
df['Feature']=df['Feature'].replace('cCRE_CTCF','CTCF')
df['Feature']=df['Feature'].replace('Alu Fraction','Alu Elements')
df['Feature']=df['Feature'].replace('LINE Fraction','LINEs')
df['Feature']=df['Feature'].replace('Delfi','Length Ratios')

print(df)

###################
### Figure 3A-E ###
###################
fig = plt.figure(layout=None,figsize=(18,14))
gs = fig.add_gridspec(
                        nrows=4, ncols=20,
                        left=0.05,right=0.95,
                        top=0.97,bottom=0.15,
                        hspace=0.75, wspace=2.5,
                        )
motifs = fig.add_subplot(gs[0,:15])
sns.despine(right=True, ax=motifs)
motifs.set_title('A', loc="left", fontweight='bold')
lengths = fig.add_subplot(gs[0,15:])
sns.despine(right=True, ax=lengths)
lengths.set_title('B', loc="left", fontweight='bold')
chromatin = fig.add_subplot(gs[1,:11])
sns.despine(right=True, ax=chromatin)
chromatin.set_title('C', loc="left", fontweight='bold')
genome = fig.add_subplot(gs[1,11:17])
sns.despine(right=True, ax=genome)
genome.set_title('D', loc="left", fontweight='bold')
aneuploidy = fig.add_subplot(gs[1,17:])
sns.despine(right=True, ax=aneuploidy)
aneuploidy.set_title('E', loc="left", fontweight='bold')

plot_dict={
    'Fragment Motif':motifs,
    'Fragment Length':lengths,
    'Breakpoint Analysis':chromatin,
    'Genomic Annotation':genome,
    'Aneuploidy':aneuploidy
    }
plots.append(fig)
for idx,type in enumerate(variable_order):
    if type == 'Delfi':continue
    ax=plot_dict[type]
    ax.set_ylim(0,0.6)
    ax.set_title(type)
    t=df[df['Variable Type']==type].sort_values(
                    'S98',ascending=False,ignore_index=True)
    sns.barplot(x=t['Feature'],y=t['S98'],ax=ax,color=bright_colors[idx])
    ax.set_xticklabels(t['Feature'].unique(),rotation=30)

motifs.set_xlabel('')
chromatin.set_xlabel('')
lengths.set_xlabel('')
lengths.set_ylabel('')
genome.set_xlabel('')
genome.set_ylabel('')
aneuploidy.set_xlabel('')
aneuploidy.set_ylabel('')

#################
### Figure 3F ###
#################
three_f = fig.add_subplot(gs[2,:8])

three_f.set_title('F', loc="left", fontweight='bold')
df['Variable Type'] = pd.Categorical(df['Variable Type'],
            categories=variable_order, ordered=True)
df=df.sort_values(by='Variable Type')
sns.stripplot(x=df['Variable Type'],y=df['S98'],ax=three_f,palette='bright')
three_f.set_ylim(0,0.6)
three_f.set_ylabel('SAS98')
three_f.set_xlabel('')
three_f.set_xticklabels(df['Variable Type'].unique(),rotation=30)
sns.despine(right=True, ax=three_f)


#################
### Figure 3G ###
#################
three_g = fig.add_subplot(gs[2,8:])
three_g.set_ylim(0,0.6)
three_g.set_title('G', loc="left", fontweight='bold')
sns.scatterplot(x=df['# Features'],
                y=df['S98'],
                hue=df['Variable Type'],
                hue_order=variable_order,
                palette='bright',
                ax=three_g
                )
three_g.set_ylabel('SAS98')
three_g.set_xlabel('Number of Variables in Set')
three_g.set_xscale('log')
sns.despine(right=True, ax=three_g)
#three_g.legend(loc='lower right',ncol=2, title="Feature Type")
h,l = three_g.get_legend_handles_labels()
three_g.legend_.remove()
three_g.legend(h,l, ncol=2,title='Variable Type',loc='lower right')
plt.setp(three_g.get_legend().get_texts(), fontsize='7')
plt.setp(three_g.get_legend().get_title(), fontsize='7')

#################
### Figure 3H ###
#################
three_h = fig.add_subplot(gs[3,:10])
three_h.set_title('H', loc="left", fontweight='bold')
sns.despine(right=True, ax=three_h)
mv=mv.sort_values('Overall S98',ignore_index=True,ascending=False)
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction/AluJ/OutsidePentamer/Length1','Five-View')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb','Wise-1')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction','Wise-1+AluFraction')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/OutsidePentamer','Wise-1+OutsidePentamer')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/Length1','Wise-1+Length-1')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluJ','Wise-1+AluJ')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction','Wise-1+AluFraction')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction/OutsidePentamer','Wise-1+AluFraction+OutsidePentamer')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction/Length1','Wise-1+AluFraction+Length-1')
mv['Feature']=mv['Feature'].replace('WiseCondorX_1Mb/AluFraction/AluJ','Wise-1+AluFraction+AluJ')
#fig,ax=plt.subplots(1,2,figsize=(14,7))
three_h.set_title('Two-View Co-MIGHT')
chosen=[
            'Wise-1',
            'Wise-1+AluFraction',
            'Wise-1+OutsidePentamer',
            'Wise-1+Length-1',
            'Wise-1+AluJ',
        ]
t=mv[mv['Feature'].isin(chosen)]
sns.barplot(x=t['Feature'],y=t['Overall S98'],ax=three_h,color=bright_colors[-1])
three_h.set_ylabel('SAS98')
three_h.set_xlabel('')
three_h.set_xticklabels(t['Feature'].unique(),rotation=30)


#################
### Figure 3I ###
#################
three_i = fig.add_subplot(gs[3,10:])
three_i.set_title('I', loc="left", fontweight='bold')
sns.despine(right=True, ax=three_i)
three_i.set_title('Multi-View Co-MIGHT')
chosen=[
            'Wise-1+AluFraction',
            'Wise-1+AluFraction+OutsidePentamer',
            'Wise-1+AluFraction+Length1',
            'Wise-1+AluFraction+AluJ',
            'Five-View',
        ]
t=mv[mv['Feature'].isin(chosen)]
sns.barplot(x=t['Feature'],y=t['Overall S98'],ax=three_i,color=bright_colors[-1])
three_i.set_ylabel('SAS98')
three_i.set_xticklabels(t['Feature'].unique(),rotation=30)
three_i.set_xlabel('')
#plt.tight_layout()
plots.append((fig,ax))

plt.savefig(f'{outdir}/Figure3.png')#, bbox_inches='tight')
plt.savefig(f'{outdir}/Figure3.svg')#, bbox_inches='tight')

################################################################################

                            ################
                            ### Figure 4 ###
                            ################

################################################################################


#-- thresholds for 98% specificity --#
tdict={'LogisticRegression':0.8083700148675955,
        'SVM':0.7343152912682109,
        'DefaultEnsembles':0.569423,
        'RandomForest':0.42949999999999966,
    }
fig,ax=plt.subplots(2,3,figsize=(14,11))
LABELS=[['A','B','C'],['D','E','F']]
three_i.set_title('I', loc="left", fontweight='bold')
for x in range(2):
    for y in range(3):
        sns.despine(right=True, ax=ax[x,y])
        ax[x,y].set_title(LABELS[x][y], loc="left", fontweight='bold')

for count,i in enumerate(['LogisticRegression','SVM','RandomForest']):
    threshold=tdict[i]
    df=pd.read_csv(f'{dir}/{i}.validation.posteriors')
    t=pd.read_csv(f'{dir}/{i}.validation.performance')
    df=df.sort_values('Posterior',ignore_index=True)
    print(i)
    print(df['Cancer Status'].value_counts())
    #-- ROC --#
    fpr,tpr,_=roc_curve(df['Cancer Status'],df['Posterior'])
    sns.lineplot(x=fpr,y=tpr,estimator=None,ax=ax[1,count],color=bright_colors[0])
    ax[1,count].set_xlabel('FPR')
    ax[1,count].set_ylabel('TPR')
    ax[0,count].set_title(i)
    ax[0,count].set_ylim(-0.05,1.05)
    axins = inset_axes(ax[1,count], width="40%", height="40%", loc=4, borderpad=2)
    fpr_inset,tpr_inset = [],[]
    for f,t in list(zip(fpr,tpr)):
        if f < 0.1:
            fpr_inset.append(f)
            tpr_inset.append(t)
    axins.plot(fpr_inset,tpr_inset,color=bright_colors[0])
    axins.set_ylim(0,0.65)
    #-- SAM Plot --#
    healthy=df[df['Cancer Status']==0].sort_values('Posterior',
                ignore_index=True,ascending=False)
    sns.scatterplot(
                x=range(len(healthy)),
                y=healthy['Posterior'],
                color=bright_colors[0],
                ax=ax[0,count],label='Healthy')
    cancer=df[df['Cancer Status']==1].sort_values('Posterior',
                ignore_index=True,ascending=False)
    sns.scatterplot(
                x=range(len(healthy),len(healthy)+len(cancer)),
                y=cancer['Posterior'],
                color=bright_colors[1],
                ax=ax[0,count],label='Cancer')
    ax[0,count].plot(
                    range(len(df)),
                    [threshold]*len(df),
                    linestyle='dashed',
                    color='r'
                    )
    ax[0,count].set_xlabel('Sample')

plt.savefig(f'{outdir}/Figure4.png')
plt.savefig(f'{outdir}/Figure4.svg')

################################################################################

                            ##################
                            ### Figure S11 ###
                            ##################

################################################################################

plots=[]
LABELS=['A','B','C']
plot_dict={}
for idx,model in enumerate(['LogisticRegression','SVM','RandomForest']):
    fig = plt.figure(layout=None,figsize=(14,5))
    fig.suptitle(f'{model}')
    if idx==2:
        fig.text(0.5, 0.04, 'Tumor Type', ha='center')
    gs = fig.add_gridspec(nrows=1, ncols=8, left=0.05, right=0.95,
                          hspace=0.15, bottom=0.2,wspace=0.00)
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title(LABELS[idx], loc="left", fontweight='bold')
    sns.despine(right=True, top=False, ax=ax0)
    ax1 = fig.add_subplot(gs[1:])
    sns.despine(left=True, right=False,top=False, ax=ax1)
    plot_dict[idx]=[ax0,ax1]
    plots.append(fig)

for idx,model in enumerate(['LogisticRegression','SVM','RandomForest']):
    df=pd.read_csv(f'{dir}/{model}.validation.posteriors')
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('A'))
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('A1'))
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('A2'))
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('A3'))
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('B'))
    df['Stage']=df['Stage'].apply(lambda x:str(x).strip('C'))
    df['Stage']=df['Stage'].replace('nan','Normal')
    df['Stage']=df['Stage'].replace('IIIV','III')
    df['Stage']=df['Stage'].replace('IIII','III')
    threshold=pd.read_csv(f'{dir}/{model}.validation.performance').iloc[2,1]
    performance=[]
    for type in df['Tumor type'].unique():
        if type=='Esophagus':continue
        t=df[df['Tumor type']==type]
        for stage in t['Stage'].unique():
            tt=t[t['Stage']==stage]
            #print(tt)
            posteriors=tt['Posterior'].to_numpy()
            y_class = (posteriors >= threshold).astype(bool)
            fp = 0
            tp = 0
            fn = 0
            tn = 0
            for true,pred in list(zip(tt['Cancer Status'],y_class)):
                if true == 1 and pred == 1:
                    tp += 1
                if true == 1 and pred == 0:
                    fn += 1
                if true == 0 and pred == 1:
                    fp += 1
                if true == 0 and pred == 0:
                    tn += 1
            try:
                sens = tp/(tp+fn)
            except:
                sens = None
            try:
                spec = tn/(tn+fp)
            except:
                spec=None
            if type == 'Healthy':
                sens = 1 - spec
            performance.append({'Tumor type':type,'Stage':stage,
                    'Observed Specificity':spec,'% Classified Positive':sens})
    performance=pd.DataFrame(performance)
    h=df[df['Tumor type']=='Healthy']
    df=df[~df['Tumor type'].isin(['Healthy','Esophagus'])]
    hp=performance[performance['Tumor type']=='Healthy']
    cp=performance[performance['Tumor type']!='Healthy']
    order=['Pancreas','Breast','Ovary','Lung',
        'Colorectal','Liver','Stomach','Kidney']
    cp['Tumor type'] = pd.Categorical(cp['Tumor type'],
                categories=order, ordered=True)
    cp=cp.sort_values(by='Tumor type')
    sns.barplot(
                x=hp['Tumor type'],
                y=hp['% Classified Positive'],
                ax=plot_dict[idx][0],
                color=bright_colors[0],
                )
    sns.barplot(
                x=cp['Tumor type'],
                y=cp['% Classified Positive'], #y=cp['Observed Sensitivity'],
                hue=cp['Stage'],
                hue_order=['I','II','III','IV'],
                palette=bright_colors[1:],
                ax=plot_dict[idx][1]
                )
    plot_dict[idx][1].tick_params(left=False,right=False,
                            labelright=False,labelleft=False)
    plot_dict[idx][0].set_ylim(0,1)
    plot_dict[idx][1].set_ylim(0,1)
    plot_dict[idx][1].set_ylabel('')
    plot_dict[idx][0].set_xlabel('')
    plot_dict[idx][1].set_xlabel('')
    if idx in [1,2]:
        plot_dict[idx][1].get_legend().remove()
    else:
        h,l = plot_dict[idx][1].get_legend_handles_labels()
        plot_dict[idx][1].legend_.remove()
        plot_dict[idx][0].legend(h,l, ncol=2,title='Stage',loc='upper left')
    plots[idx].savefig(f'{outdir}/FigureS8{LABELS[idx]}.png')
    plots[idx].savefig(f'{outdir}/FigureS8{LABELS[idx]}.svg')
