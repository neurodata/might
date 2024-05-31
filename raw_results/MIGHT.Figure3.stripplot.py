import numpy as np
import itertools
from matplotlib.gridspec import SubplotSpec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_palette("bright")
sns.set(font_scale=1.1)
color_palette = sns.color_palette(palette='Set1')
sns.set_style('white')

variable_order=[
            'Fragment Length',
            'Fragment Motif',
            'Breakpoints',
            'Genomic Annotation',
            'Aneuploidy']


results='/data/kvhdata1/Projects/MIGHT/Manuscript/Figures'
plots=[]


################################################################################

                            ################
                            ### Figure 3 ###
                            ################

################################################################################

dir='/data/kvhdata1/Projects/MIGHT/results/Cohort1/TestingTreesAndMaxFeatures/results'
df=pd.read_csv('/data/kvhdata1/Projects/MIGHT/Manuscript/Tables/supptable2.csv')
df['Variable Type']=df['Variable Type'].replace('DyadAccessibility','Breakpoints')
df['id']=df['Variable Type']+df['Variable Set 1']
d1=df[df['Type of MIGHT']=='MIGHT']

vdict={a:b for a,b in list(zip(d1['Variable Set 1'],d1['Variable Type']))}
adict={a:b for a,b in list(zip(d1['Variable Set 1'],d1['Abbreviation']))}
d2=df[df['Variable Type']=='TwoView']
d3=df[df['Variable Type']=='ThreeView']

###################
### Figure 5A-E ###
###################
fig = plt.figure(layout=None,figsize=(25,20))
gs = fig.add_gridspec(
                        nrows=9, ncols=20,
                        left=0.05,right=0.97,
                        top=0.97,bottom=0.05,
                        hspace=2.1, wspace=2.5,
                        )
#-- 5A --#
lengths = fig.add_subplot(gs[0:2,:5])
sns.despine(right=True, ax=lengths)
lengths.set_title('A', loc="left", fontsize=18,fontweight='bold')
#-- 5B --#
motifs = fig.add_subplot(gs[0:2,5:])
sns.despine(right=True, ax=motifs)
motifs.set_title('B', loc="left", fontsize=18,fontweight='bold')
#-- 5C --#
chromatin = fig.add_subplot(gs[2:4,:11])
sns.despine(right=True, ax=chromatin)
chromatin.set_title('C', loc="left", fontsize=18,fontweight='bold')
#-- 5D --#
genome = fig.add_subplot(gs[2:4,11:16])
sns.despine(right=True, ax=genome)
genome.set_title('D', loc="left", fontsize=18,fontweight='bold')
#-- 5E --#
aneuploidy = fig.add_subplot(gs[2:4,16:])
sns.despine(right=True, ax=aneuploidy)
aneuploidy.set_title('E', loc="left", fontsize=18,fontweight='bold')

plot_dict={
    'Fragment Length':lengths,
    'Fragment Motif':motifs,
    'Breakpoints':chromatin,
    'Genomic Annotation':genome,
    'Aneuploidy':aneuploidy
    }
plots.append(fig)

mv_order=[]
for idx,type in enumerate(variable_order):
    if type == 'Delfi':continue
    ax=plot_dict[type]
    ax.set_ylim(0,0.65)
    ax.set_title(type,fontsize=22)
    t=d1[d1['Variable Type']==type].sort_values(
                    'S@98',ascending=False,ignore_index=True)
    print(t)
    sns.barplot(x=t['Variable Set 1'],
        y=t['S@98'],
        ax=ax,
        color=color_palette[idx])
    mv_order.extend(t['id'].tolist())
    ax.set_ylabel('S@98',fontsize=16)
    ax.set_xticklabels(t['Abbreviation'].unique(),fontsize=18,rotation=30)
    ax.tick_params(axis='y', labelsize=18)

lengths.set_xlabel('')
motifs.set_xlabel('')
motifs.set_ylabel('')
chromatin.set_xlabel('')
genome.set_xlabel('')
genome.set_ylabel('')
aneuploidy.set_xlabel('')
aneuploidy.set_ylabel('')

#################
### Figure 5F ###
#################
lst=[]
for variable in d1['Variable Set 1'].unique():
    a=d1[d1['Variable Set 1']==variable]['S@98'].tolist()[0]
    b=d2[d2['Variable Set 1']==variable]
    #b['Log Fold Change in S@98 (%)']=np.log(b['S@98']/a)
    b['Change in S@98 (%)']=100*(b['S@98']-a)
    lst.append(b)

d2=pd.concat(lst)
d2['Variable Type']=d2['Variable Set 1'].apply(lambda x:vdict[x])
d2['Variable Class of 2nd View']=d2['Variable Set 2'].apply(lambda x:vdict[x])
d2['Abbreviation']=d2['Variable Set 1'].apply(lambda x:adict[x])
d2['id']=d2['Variable Type']+d2['Variable Set 1']
d2['id'] = pd.Categorical(d2['id'],
            categories=mv_order, ordered=True)
d2=d2.sort_values(by='id')

five_f = fig.add_subplot(gs[4:7,:])
five_f.set_title('F', loc="left", fontsize=18,fontweight='bold')
sns.despine(right=True, ax=five_f)
sns.stripplot(
                x=d2['Variable Set 1'],
                #y=d2['Change in S@98 (%)'],
                y=d2['S@98'],
                hue=d2['Variable Class of 2nd View'],
                hue_order=variable_order,
                palette=color_palette,
                ax=five_f
                )

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

handles, labels = five_f.get_legend_handles_labels()
plt.legend(
            flip(handles, 3),
            flip(labels, 3),
            ncol=3,
            fontsize=16,
            title='Variable Class of 2nd View',
            title_fontsize=16,
        )
five_f.set_xticklabels(
        [adict[a] for a in d2['Variable Set 1'].unique()],
        rotation=60,
        fontsize=16)
five_f.set_xlabel('')
#five_f.set_ylabel('Change in S@98 (%)',fontsize=16)
five_f.set_ylabel('S@98',fontsize=16)
five_f.plot([0,42],[0,0],color='k',linestyle='--')
'''five_f.plot([5.5,5.5],[-25,50],color='k',linestyle='--')
five_f.plot([24.5,24.5],[-25,50],color='k',linestyle='--')
five_f.plot([35.5,35.5],[-25,50],color='k',linestyle='--')
five_f.plot([39.5,39.5],[-25,50],color='k',linestyle='--')'''
five_f.plot([5.5,5.5],[0,0.65],color='k',linestyle='--')
five_f.plot([24.5,24.5],[0,0.65],color='k',linestyle='--')
five_f.plot([35.5,35.5],[0,0.65],color='k',linestyle='--')
five_f.plot([39.5,39.5],[0,0.65],color='k',linestyle='--')
five_f.set_xlim(-0.5,42.5)
five_f.tick_params(axis='y', labelsize=18)



#################
### Figure 5G ###
#################
a=d1[d1['Variable Set 1']=='Wise-1']['S@98'].tolist()[0]
b=d2[d2['Variable Set 1']=='Wise-1']
#b['Relative Change in S@98 (%)']=(100*b['S@98']/a)-100
b['Change in S@98 (%)']=100*(b['S@98']-a)
#b['Added Information?'] = (b['Relative Change in S@98 (%)'] >= 0).astype(bool)
b['Increased S@98?'] = (b['Change in S@98 (%)'] >= 0).astype(bool)
b=b.sort_values('Change in S@98 (%)')

five_g = fig.add_subplot(gs[7:,:])
five_g.set_title('G', loc="left", fontsize=18,fontweight='bold')
sns.despine(right=True, ax=five_g)
sns.barplot(
            x=b['Variable Set 2'],
            y=b['Change in S@98 (%)'],
            hue=b['Increased S@98?'],
            ax=five_g,
            )

five_g.set_xticklabels([])
five_g.set_xlabel('')
five_g.set_ylabel('Change in S@98 (%)',fontsize=16)
five_g.tick_params(axis='y', labelsize=18)

plt.savefig(f'{results}/Figure3.strip.png')#, bbox_inches='tight')
plt.savefig(f'{results}/Figure3.strip.svg')#, bbox_inches='tight')
