#!/usr/bin/env python
# coding: utf-8

# NOTE: This was converted from a Jupyter notebook using nbconvert, and requires the ipython kernel installed on a UNIX-like system with access to bash to execute system commands. System commmands are executed with get_ipython().system("command")

# # What is the distribution of paper lengths in CSCW?
# by R. Stuart Geiger ([@staeiou](http://twitter.com/staeiou)), staff ethnographer, [Berkeley Institute for Data Science](http://bids.berkeley.edu)
# 
# Freely licensed under both [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [The MIT License](https://opensource.org/licenses/MIT).
# 
# This is the data analysis and visualization notebook, which builds off the `cscw-pages-notext.csv` created in `data-cleaning-processing.ipynb`

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as ticker

import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as po

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)
#pd.set_option('display.max_rows', -1)


# ## Data import and processing
# 

# In[2]:


merged_df = pd.read_csv("../data/cscw-pages-notext.csv", index_col=0)


# In[3]:


merged_df = merged_df.query("year != 2019")


# In[4]:


merged_df.sample(3).T


# This `year_str` is because I haven't figured out how to get some visualization libraries like `plotly` to plot years as separate categories. It tries to be smart and plot it on a time axis.

# In[5]:


def year_str_convert(year):
    if year == 2017.5 or year == '2017.5':
        return '* 2017.5'
    else:
        return "* " + str(year)[0:4] 


# In[6]:


merged_df['year_str'] = merged_df['year'].apply(year_str_convert)


# In[7]:


merged_df['year_str'].value_counts().sort_index()


# In[8]:


years = list(merged_df['year_str'].value_counts().index)
years.sort()


# ## Descriptive statistics
# 

# ### Presented in the paper
# 
# "The longest paper published in CSCW before 2013 had a main section length of 10,578 words."

# In[9]:


(merged_df.query("year_float < 2013 & num_pages > 4").body_len_words).max()


# "The upper 95th percentile length for 2000-2012 non-note papers (> 4-pages) was 9,509 words."

# In[10]:


(merged_df.query("year_float < 2013 & num_pages > 4").body_len_words).quantile(.95)


# "In the combined past two PACMHCI rounds, 51.0% of all papers had a main section length longer than the longest pre-2013 CSCW paper."

# In[11]:


len(merged_df.query("year_float > 2017 & body_len_words > 10578")) / len(merged_df.query("year_float > 2017"))


# "73.4\% of all PACMHCI papers had a main section length longer than the upper 95th percentile length for 2000-2012 non-note papers."

# In[12]:


len(merged_df.query("year_float > 2017 & body_len_words > 9509")) / len(merged_df.query("year_float > 2017"))


# ### Exploratory / not presented in the paper

# Descriptive statistics for non-notes before 2013

# In[13]:


(merged_df.query("year_float < 2013 & num_pages > 4").body_len_words).describe()


# Descriptive statistics for notes before 2013

# In[14]:


(merged_df.query("year_float < 2013 & num_pages == 4").body_len_words).describe()


# Descriptive statistics for all papers after 2012

# In[15]:


(merged_df.query("year_float > 2012").body_len_words).describe()


# Descriptive statistics for all papers 2013-2017

# In[16]:


(merged_df.query("year_float > 2012 & year_float < 2017.5").body_len_words).describe()


# Descriptive statistics for all papers in PACMHCI (2017.5 and 2018)

# In[17]:


(merged_df.query("year_float > 2017").body_len_words).describe()


# What proportion of PACMHCI CSCW papers are longer than the upper median length paper published before 2013?

# In[18]:


len(merged_df.query("year_float > 2017 & body_len_words > 7753")) / len(merged_df.query("year_float > 2017"))


# ## Visualizations on word lengths
# 
# ### Figure 1: Boxplot + stripplot for total number of words (incl. front matter, references, appendices)

# In[19]:


sns.set(font_scale=1.05, style='whitegrid')

fig, ax = plt.subplots(figsize=[12,4])

scatter = sns.stripplot(data=merged_df,
            y='words',
            x='year',
            jitter=True,
            s=3,
            ax=ax
           )

sns.boxplot(data=merged_df,
            y='words',
            x='year',
            color=".75",
            fliersize=0,
            linewidth=1.15,
            whis=[5,95],
            ax=ax
           )

plt.title("Length (in words) of CSCW papers (including references and appendices) over time")

ax.set_ylabel("words", fontweight='bold')
ax.set_xlabel("year", fontweight='bold')

# Create 3 lines that segment the 4 major periods

plt.axvline(1.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)
plt.axvline(7.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)
plt.axvline(12.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)

# Color the background grey for the first and third sections 

plt.axvspan(-2, 1.5, facecolor='.1', alpha=0.09, zorder=-100)
plt.axvspan(7.5, 12.5, facecolor='.1', alpha=0.09, zorder=-100)

# Label the sections

plt.text(-.4,600,"Pre-note era")
plt.text(3.5,600,"Offical note era")
plt.text(8.5,600,"No page restriction era")
plt.text(12.6,600,"PACMHCI era")

plt.ylim(0,22500)

ax.xaxis.set_ticklabels([2000,2002,2004,2006,2008,2010,2011,2012,2013,2014,2015,2016,2017,2017.5,2018])

plt.savefig("../figures/fig1-word-len-all.pdf", bbox_inches='tight', dpi=300)


# ### Figure 2: Boxplot + stripplot for number of words in the main body + front matter (no references or appendices)

# In[20]:


sns.set(font_scale=1.05, style='whitegrid')
fig, ax = plt.subplots(figsize=[12,4])

sns.stripplot(data=merged_df,
            y='body_len_words',
            x='year',
            jitter=True,
            s=3,
            ax=ax
           )

sns.boxplot(data=merged_df,
            y='body_len_words',
            x='year',
            color=".75",
            fliersize=0,
            linewidth=1.15,
            whis=[5,95],
            ax=ax
           )

ax.set_ylabel("words", fontweight='bold')
ax.set_xlabel("year", fontweight='bold')

# Create 3 lines that segment the 4 major periods

plt.axvline(1.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)
plt.axvline(7.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)
plt.axvline(12.5, ymin=0, zorder=100, clip_on=False, color='k', alpha=.5, linewidth=.75)

# Color the background grey for the first and third sections 

plt.axvspan(-2, 1.5, facecolor='.1', alpha=0.09, zorder=-100)
plt.axvspan(7.5, 12.5, facecolor='.1', alpha=0.09, zorder=-100)

# Label the sections

plt.text(-.4,600,"Pre-note era")
plt.text(3.5,600,"Offical note era")
plt.text(8.5,600,"No page restriction era")
plt.text(12.6,600,"PACMHCI era")

plt.ylim(0,20000)

ax.xaxis.set_ticklabels([2000,2002,2004,2006,2008,2010,2011,2012,2013,2014,2015,2016,2017,2017.5,2018])

plt.title("Length (in words) of CSCW paper main sections (no references or appendices) over time")
plt.savefig("../figures/fig2-word-len-body.pdf", bbox_inches='tight', dpi=300)


# # Distribution of paper lengths by somewhat arbitrary categories
# ## Exploratory
# ### Histograms to identify break points
# 
# #### Distributions overall

# In[21]:


merged_df.body_len_words.hist(bins=50,figsize=[10,4])


# #### Distributions in the official notes era

# In[22]:


merged_df.query("2004 < year_float < 2013").body_len_words.hist(bins=50,figsize=[10,4])


# In[23]:


merged_df.query("num_pages == 4").body_len_words.describe()


# ### Defining length functions

# In[24]:


def is_note(row):
    #print(row)
    if float(row['words']) < 5000:
        return True
    else:
        return False


# In[25]:


def paper_type(row):
    
    num_words = float(row['words'])
    
    if num_words < 2500:
        return 'abstract'
    elif 5000 > num_words > 2500:
        return 'note'
    elif 11000 > num_words > 5000:  
        return '10 pager'
    elif 16000 > num_words > 11000:
        return 'journal article'
    else:
        return 'long read'


# In[26]:


def paper_type_alt(row):
    
    num_words = float(row['body_len_words'])
    
    if num_words < 4500:
        return 'note'
    elif 7500 > num_words >= 4500:
        return 'short paper'
    elif 10500 > num_words >= 7500:  
        return 'full paper'
    elif 15000 > num_words >= 10500:
        return 'journal article'
    elif num_words > 15000:
        return 'mega article'


# In[27]:


merged_df['is_note'] = merged_df.apply(is_note, axis=1)
merged_df['paper_type'] = merged_df.apply(paper_type, axis=1)
merged_df['paper_type_alt'] = merged_df.apply(paper_type_alt, axis=1)


# In[ ]:





# ### Analysis on `paper_type_alt` function
# 
# In the paper, I ended up using the `paper_type_alt` function

# In[28]:


merged_df.paper_type_alt.value_counts()


# In[29]:


df_types_count = merged_df.groupby('year').             paper_type_alt.value_counts(normalize=False, sort=True).unstack()
df_types_count


# "From 2004 to 2012, there was a consistent cluster of notes and longer papers, with the proportion of notes ranging from 21\% (2011) to 38\% (2010)."

# In[30]:


df_types = merged_df.groupby('year').             paper_type_alt.value_counts(normalize=True, sort=True).unstack()
df_types


# In[31]:


df_types = df_types[['note', 'short paper', 'full paper', 'journal article', 'mega article']]


# ### Figure 3: Distribution of somewhat arbitrary categories of main body length

# In[32]:


sns.set(font_scale=1.2, style="whitegrid")
pal = sns.color_palette("colorblind")

ax = df_types.plot(kind='bar', stacked=True, figsize=[13,4], width=.85, color=pal)
legend_labels = ['Note (< 4,500 words)', 'Short paper (4,500-7,500 words)', 'Full paper (7,500-10,500 words)', 'Journal article (10,500-15,000 words)', 'Mega article (>15,000 words)']
ax.legend(bbox_to_anchor=(1.0,.95), labels=legend_labels)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax.set_ylabel("% of papers in category")
plt.title("Distribution of CSCW papers by length of main body (no references or appendices) in words",
         fontdict={'fontsize': 17})
#plt.axvspan(-2, 1.5, facecolor='.1', alpha=0.3, zorder=-100)
#plt.axvspan(7.5, 12.45, facecolor='.1', alpha=0.3,zorder=-100)
ax.set_xlabel("year", fontweight='bold', labelpad=30)

plt.text(-.5,-0.2,"Pre-note era")
plt.text(3,-0.2,"Offical note era")
plt.text(8.3,-0.2,"No page restriction era")
plt.text(12.7,-0.2,"PACMHCI era")
#plt.ylim(-.17,1.1)
#plt.ylim(0,20000)
ax.xaxis.set_ticklabels([2000,2002,2004,2006,2008,2010,2011,2012,2013,2014,2015,2016,2017," 2017.5",2018])
plt.xticks(rotation='horizontal')

plt.axhline(-.13,  zorder=100, clip_on=False, color='k', alpha=.5, linewidth=1)


plt.axvline(1.5, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)
plt.axvline(7.5, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)
plt.axvline(12.45, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)
plt.ylim(0,1.05)
plt.savefig("../figures/fig3-dist-len-cat.pdf", bbox_inches='tight', dpi=300)


# ## Interactive/web visualizations

# In[33]:


merged_df['filename'] = merged_df.index


# In[34]:


fig = px.box(merged_df,y='body_len_words',
             x='year_str', hover_data=['title_from_text','lead_author'],
            points='all',range_x=[-1,15], title="CSCW main section paper length (no references or appendixes) by year")

fig.update_layout()

fig.show()



# In[35]:


po.plot(fig, filename="../figures/len_by_year.html")


# In[36]:


fig = go.Figure(data=[go.Table(
                columnwidth = [10,50,15,10,10,10,10],
                header=dict(values=['Year', 'Paper title','Lead Author', 'Pages', 'Main body length (words)', 'References length (words)', 'Appendix length (words)', 'Total length (words)'],
                            align='left'),
                cells=dict(values=[merged_df['year'], merged_df['title_from_text'], merged_df['lead_author'], merged_df['num_pages'], merged_df['body_len_words'], merged_df['ref_len_words'], merged_df['appx_len_words'], merged_df['words']],
                            align='left'))
])

fig.show()

po.plot(fig, filename="../figures/web_table.html")


# ### Proportion of papers that have quotes in the title

# In[37]:


df_quote_title = merged_df.groupby('year')['title_has_quote'].value_counts(normalize=False, sort=True).unstack()
df_quote_title


# In[38]:


ax = df_quote_title.plot(kind='bar', stacked=True, figsize=[13,4], width=.85, color=pal)
ax.xaxis.set_ticklabels([2000,2002,2004,2006,2008,2010,2011,2012,2013,2014,2015,2016,2017," 2017.5",2018])
legend_labels = ['No quotes in title', 'Quotes in title']
ax.legend(bbox_to_anchor=(1.0,.95), labels=legend_labels)
vals = ax.get_yticks()
#ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax.set_ylabel("Number of papers")
plt.title("Distribution of CSCW papers by presence or absence of quotes in titles",
         fontdict={'fontsize': 17})
#plt.axvspan(-2, 1.5, facecolor='.1', alpha=0.3, zorder=-100)
#plt.axvspan(7.5, 12.45, facecolor='.1', alpha=0.3,zorder=-100)
ax.set_xlabel("year", fontweight='bold', labelpad=30)

plt.text(-.5,-36.2,"Pre-note era")
plt.text(3,-36.2,"Offical note era")
plt.text(8.3,-36.2,"No page restriction era")
plt.text(12.7,-36.2,"PACMHCI era")
#plt.ylim(-.17,1.1)
#plt.ylim(0,20000)
ax.xaxis.set_ticklabels([2000,2002,2004,2006,2008,2010,2011,2012,2013,2014,2015,2016,2017," 2017.5",2018])
plt.xticks(rotation='horizontal')

#plt.axhline(-60,  zorder=100, clip_on=False, color='k', alpha=.5, linewidth=1)


plt.axvline(1.5, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)
plt.axvline(7.5, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)
plt.axvline(12.52, ymin=-.2, zorder=100, clip_on=False, color='k', alpha=.5)

plt.savefig("../figures/quotes-in-titles.pdf", bbox_inches='tight', dpi=300)


# In[ ]:




