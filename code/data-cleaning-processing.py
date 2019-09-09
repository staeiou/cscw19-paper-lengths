#!/usr/bin/env python
# coding: utf-8

# NOTE: This was converted from a Jupyter notebook using nbconvert, and requires the ipython kernel installed on a UNIX-like system with access to bash to execute system commands. System commmands are executed with get_ipython().system("command")

# # Data processing for "The Rise and Fall of the Note: Changing Paper Lengths in ACM CSCW, 2000-2018"
#
# by R. Stuart Geiger ([@staeiou](http://twitter.com/staeiou)), staff ethnographer, [Berkeley Institute for Data Science](http://bids.berkeley.edu)
#
# Freely licensed under both [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [The MIT License](https://opensource.org/licenses/MIT).

# This notebook does almost all of the data processing for the study. Paper PDFs were manually downloaded into `../data/pdfs/YEAR/`, then batch converted to .docx using Adobe Acrobat DC, which are found in `../data/docx/YEAR/`. I tried a number of different approaches using free/open-source software, but due to the age of many of the PDFs in the dataset and Adobe's effective ownership of the PDF format, this was the best way to get consistent results. 
#
# This notebook converts all the .docx files to .txt using `pandoc`, loads the plain text and other metadata into a pandas dataframe, and extracts reference and appendix sections. Two data files are output: `cscw-pages-all.csv` includes the full plain text and is not publicly shared, but `cscw-pages-notext.csv` only contains quantitative metrics. These are analyzed by `analysis-viz.ipynb`. 
#
# Note that this notebook also includes one paper from 2019, the camera-ready version for this study, just for comparison purposes. This paper is not included in the dataset that is used for data analysis and visualization.
#
# I am using a mix of bash commands and python scripts to process and collect the data. Various standard GNU/Linux tools are used that may or may not be available on other OSes, including: `wc`, `find`, `xargs`, as well as `pandoc` (which is not commonly included in Linux OSes). This notebook also uses`imagemagick` via the `Wand` python connector library to display images of PDFs in this notebook for demonstration purposes, but these are not a core requirement of the data processing pipeline.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import datetime
from wand.image import Image as WImage

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)


# In[2]:


start = datetime.datetime.now()


# ## Number of pages

# In[3]:


import glob
from PyPDF2 import PdfFileReader


# In[4]:


years = list(range(2010,2019))
years.append(2000)
years.append(2002)
years.append(2004)
years.append(2006)
years.append(2008)
years.append(2017.5)
years.append(2019)
years.sort()
years


# In[5]:


pages_list = []
for year in years:
    print(year)
    for pdf in glob.glob("../data/pdfs/" + str(year) + "/*.pdf"):
        pdf_obj = PdfFileReader(open(pdf,'rb'), strict=False)
        num_pages = pdf_obj.getNumPages()
        orientation = pdf_obj.getPage(0).get('/Rotate')
        row = {'filename':pdf[13:-4],'year':year,'num_pages':num_pages,'orientation':orientation}
        pages_list.append(row)


# In[6]:


df_pages = pd.DataFrame(pages_list).set_index('filename')
df_pages[0:5]


# In[7]:


gb = df_pages.groupby('year')
pd.DataFrame(gb['num_pages'].value_counts(sort=False))
#df_pages.num_pages.value_counts()


# In[8]:


import matplotlib.pyplot as plt
sns.set(font_scale=1.25)
fig, ax = plt.subplots(figsize=[14,8])
sns.boxplot(data=df_pages.dropna().query("orientation == 0"), y='num_pages', x='year', whis=[5,95], ax=ax)
plt.suptitle("Boxplots for length of CSCW papers over time")


# In[9]:


end = datetime.datetime.now()

time_to_run = end - start
minutes = int(time_to_run.seconds/60)
seconds = time_to_run.seconds % 60
print("Current runtime: ", minutes, "minutes, ", seconds, "seconds")


# ## Process docx to text

# In[10]:


get_ipython().system('unzip -o ../data/docx/corrected.zip -d ../data/docx')


# The following bash command converts all the .docx files to .txt using `pandoc` and `xargs` to parallelize. The number of threads is set with the -P flag, you might want to change from -P8 to the number of CPUs you have.

# In[11]:


get_ipython().system('find ../data/docx/ -name "*.docx" -print0 | xargs -0 -n2 -P8 -I{} pandoc {} -t plain -o {}.txt')


# In[12]:


end = datetime.datetime.now()

time_to_run = end - start
minutes = int(time_to_run.seconds/60)
seconds = time_to_run.seconds % 60
print("Current runtime: ", minutes, "minutes, ", seconds, "seconds")


# In[13]:


# This command replaces all punctuation with spaces, which I ended up not choosing to do.

#!find ../data/docx/ -name "*.docx.txt" -exec sed -i 's/[[:punct:]]/ /g' {} \;


# In[14]:


get_ipython().system('rm -rf ../data/txt')
get_ipython().system('mkdir ../data/txt')


# In[15]:


get_ipython().system('cp -r ../data/docx/* ../data/txt/')


# In[16]:


get_ipython().system('find ../data/txt/ -name "*.docx" -type f -delete ')


# The following bash command runs `wc` to get word counts for all papers, although these ended up being discarded for a more consistent word count in python (which does counts for the entire paper and the various sections of the paper). However, this is kept because it is used for indexing. The `head -n -1` cuts off the last line of the `wc` output, which prints a total when there are multiple files. Since there is only one 2019 paper here (this one!), this has to be run separately.

# In[17]:


get_ipython().run_cell_magic('bash', '', "rm -rf ../data/word_counts/\nmkdir ../data/word_counts/\nfor year in {2000,2002,2004,2006,2008,{2010..2017},2017.5,2018}\ndo\n    wc -w ../data/txt/$year/*.docx.txt | sed 's/^ *//' | head -n -1 > ../data/word_counts/cscw-$year-pages.csv\ndone\n\nwc -w ../data/txt/2019/*.docx.txt | sed 's/^ *//' > ../data/word_counts/cscw-2019-pages.csv")


# In[18]:


get_ipython().system('ls ../data/word_counts')


# In[19]:


get_ipython().system('head ../data/word_counts/cscw-2000-pages.csv')


# In[20]:


get_ipython().system('head ../data/word_counts/cscw-2018-pages.csv')


# ## Merge dataframes, spot check

# In[21]:


df_words = pd.DataFrame(columns=["filename","year", "words"])

for year in years:
    df_year = pd.read_csv("../data/word_counts/cscw-"+ str(year) + "-pages.csv", sep=" ", names=["words", "filename"])
    
    for idx, row in df_year.iterrows():
        df_words = df_words.append({"filename":row['filename'][12:-9], "year": str(year), "words":row['words']}, ignore_index=True)


# In[22]:


df_words.words = df_words.words.astype(int)


# In[ ]:





# In[23]:


gb = df_words.groupby("year")
pd.DataFrame(gb.describe())


# In[24]:


df_words = df_words.set_index('filename')


# In[25]:


df_words['year_float'] = df_words['year'].astype(float)


# In[26]:


df_words[0:5]


# In[27]:


df_words.query("year>'2012'").sort_values('words')


# ### Character counts

# In[28]:


get_ipython().run_cell_magic('bash', '', "rm -rf ../data/char_counts/\nmkdir ../data/char_counts/\nfor year in {2000,2002,2004,2006,2008,{2010..2017},2017.5,2018}\ndo\n    wc -m ../data/txt/$year/*.docx.txt | sed 's/^ *//' | head -n -1 > ../data/char_counts/cscw-$year-pages.csv\ndone\n\nwc -m ../data/txt/2019/*.docx.txt | sed 's/^ *//' > ../data/char_counts/cscw-2019-pages.csv")


# In[29]:


get_ipython().system('ls ../data/char_counts')


# In[30]:


get_ipython().system('head ../data/char_counts/cscw-2000-pages.csv')


# In[31]:


df_chars = pd.DataFrame(columns=["filename","year", "characters"])

for year in years:
    df_year = pd.read_csv("../data/char_counts/cscw-"+ str(year) + "-pages.csv", sep=" ", names=["characters", "filename"])
    
    for idx, row in df_year.iterrows():
        df_chars = df_chars.append({"filename":row['filename'][12:-9], "year": str(year), "characters":row['characters']}, ignore_index=True)


# In[32]:


df_chars.characters = df_chars.characters.astype(int)


# In[ ]:





# In[33]:


gb = df_chars.groupby("year")
pd.DataFrame(gb.describe())


# In[34]:


df_chars = df_chars.set_index('filename')


# In[35]:


df_chars[0:5]


# ### Merge words and chars dataframes 

# In[36]:


df_words[0:5], df_chars[0:5]


# In[37]:


merged_df1 = df_words.join(df_chars, lsuffix='l_').drop('yearl_',axis=1)
merged_df1[0:5]


# In[38]:


df_pages[0:5]


# In[39]:


merged_df = merged_df1.join(df_pages, lsuffix='l_').drop('yearl_',axis=1)
merged_df[0:5]


# In[40]:


merged_df['words_per_page_total'] = merged_df['words']/merged_df['num_pages']
merged_df['chars_per_word_total'] = merged_df['characters']/merged_df['words']

merged_df['year'] = merged_df['year'].astype(str)
merged_df['year'] = merged_df['year'].str.replace('.0', '', regex=False)


# In[41]:


merged_df = merged_df[merged_df['num_pages'] > 2]
merged_df = merged_df[merged_df['words'] > 200]
merged_df[0:5]


# In[42]:


end = datetime.datetime.now()

time_to_run = end - start
minutes = int(time_to_run.seconds/60)
seconds = time_to_run.seconds % 60
print("Current runtime: ", minutes, "minutes, ", seconds, "seconds")


# In[ ]:





# ## Get reference and appendix sections

# ### Get full text in dataframe

# In[43]:


text_list = []
for year in years:
    #print(year)
    for txt in glob.glob("../data/txt/" + str(year) + "/*.txt"):
        with open(txt,"r") as file_obj:
            paper_text = file_obj.read()
        
        row = {'filename':txt[12:-9],'paper_text':paper_text}
        text_list.append(row)

df_text = pd.DataFrame(text_list).set_index('filename')


# In[44]:


df_text[0:5]


# In[45]:


merged_df = merged_df.join(df_text)
merged_df[0:5]


# In[46]:


def find_all(needle,haystack, flags):
    return [a.start() for a in list(re.finditer(needle, haystack, flags))]


# In[47]:


def get_ref_section_start(row):
    text = row['paper_text'].lower()
    regex = '^[^a-zA-Z]*(bibliography|references|reference|works cited|refefences)[^a-zA-Z]*$'
    appx_sections = find_all(regex, text, re.IGNORECASE | re.MULTILINE)
    #print(ref_sections)
    if len(appx_sections) == 1:
        return appx_sections[0]
    elif len(appx_sections) > 1:
        return appx_sections[-1]
    else:
        return False


# In[48]:


def get_appx_section_start(row):
    text = row['paper_text'].lower()
    regex = '^[^a-zA-Z]*(appendix|appendices|appendixes)[^a-zA-Z]*$'
    appx_sections = find_all(regex, text, re.IGNORECASE | re.MULTILINE)
    #print(ref_sections)
    if len(appx_sections) == 1:
        return appx_sections[0]
    elif len(appx_sections) > 1:
        return appx_sections[-1]
    else:
        return False


# In[49]:


df_appx_start = pd.DataFrame(merged_df.apply(get_appx_section_start, axis=1),
                            columns=['appx_start'])
df_ref_start = pd.DataFrame(merged_df.apply(get_ref_section_start, axis=1),
                            columns=['ref_start'])


# In[50]:


df_appx_start.sort_values(by='appx_start', ascending=False)[0:5]


# In[51]:


df_ref_start[0:5]


# In[52]:


merged_df = merged_df.merge(df_appx_start, left_index=True, right_index=True)
merged_df = merged_df.merge(df_ref_start, left_index=True, right_index=True)


# In[53]:


def get_appx_text(row):
    appx_start = row['appx_start']
    ref_start = row['ref_start']
    paper_text = row['paper_text']
    
    if appx_start is False:
        return ""
    elif appx_start > ref_start:
        return paper_text[appx_start:]
    else:
        return paper_text[appx_start:ref_start]


# In[54]:


def get_ref_text(row):
    appx_start = row['appx_start']
    ref_start = row['ref_start']
    paper_text = row['paper_text']
    
    if appx_start is False:
        return paper_text[ref_start:]
    elif appx_start > ref_start:
        return paper_text[ref_start:appx_start]
    else:
        return paper_text[ref_start:]


# #### Testing
# First, for a paper with the appendix before the references:

# In[55]:


get_appx_text(merged_df.loc['2018/cscw124-miller-hillberg'])[0:1000]


# In[56]:


get_ref_text(merged_df.loc['2018/cscw124-miller-hillberg'])[0:1000]


# Then for a paper with the appendix after the references:

# In[57]:


get_appx_text(merged_df.loc['2015/p218-azaria'])[0:1000]


# In[58]:


get_ref_text(merged_df.loc['2015/p218-azaria'])[0:1000]


# And finally, a paper with no appendix:

# In[59]:


get_ref_text(merged_df.loc['2013/p295-choi'])[0:1000]


# In[60]:


get_appx_text(merged_df.loc['2013/p295-choi'])


# ## Length calculations

# In[61]:


def word_count(text):
    return len(text.split())


# In[62]:


merged_df['appx_text'] = merged_df.apply(get_appx_text, axis=1)
merged_df['ref_text'] = merged_df.apply(get_ref_text, axis=1)

merged_df['appx_len_chars'] = merged_df['appx_text'].apply(len)
merged_df['ref_len_chars'] = merged_df['ref_text'].apply(len)

merged_df['appx_len_words'] = merged_df['appx_text'].apply(word_count)
merged_df['ref_len_words'] = merged_df['ref_text'].apply(word_count)


# ### Secondary length calculations

# In[63]:


merged_df['words_per_page'] = merged_df['words'] / merged_df['num_pages']

merged_df['body_len_chars'] = merged_df['characters'] - merged_df['appx_len_chars'] - merged_df['ref_len_chars']
merged_df['body_len_words'] = merged_df['words'] - merged_df['appx_len_words'] - merged_df['ref_len_words']

merged_df['appx_prop_words'] = merged_df['appx_len_words'] / merged_df['words']
merged_df['ref_prop_words'] = merged_df['ref_len_words'] / merged_df['words']

merged_df['appx_prop_chars'] = merged_df['appx_len_chars'] / merged_df['characters']
merged_df['ref_prop_chars'] = merged_df['ref_len_chars'] / merged_df['characters']

merged_df['body_words_per_char'] = merged_df['body_len_chars'] / merged_df['body_len_words']
merged_df['ref_words_per_char'] = merged_df['ref_len_chars'] / merged_df['ref_len_words']
merged_df['appx_words_per_char'] = merged_df['appx_len_chars'] / merged_df['appx_len_words']


# In[64]:


merged_df[['ref_len_chars','ref_len_words',
           'ref_prop_words','ref_prop_chars', 'ref_words_per_char',
           'appx_len_chars','appx_len_words', 'body_words_per_char',
           'appx_prop_words','appx_prop_chars', 'appx_words_per_char']] \
            .sort_values('appx_len_chars',ascending=False)[0:10]


# ## Remove extended abstracts and panels

# ### Remove 2004/p122-keisler
# 
# This looks like a note, but if you read it, it is longer panel description in the double-column ACM format that won't be caught through any of the methods below.

# In[65]:


WImage(filename="../data/pdfs/2004/p122-kiesler.pdf")


# In[66]:


print(merged_df.loc['2004/p122-kiesler']['paper_text'][550:1350])


# In[67]:


merged_df.drop(labels=['2004/p122-kiesler'], inplace=True)


# ### Remove extended abstracts, mostly from CSCW 2011
# 
# Panel descriptions in the extended abstract format have substantially fewer words per page. These are usually in landscape format (orientation = 90), but some are also in portrait (orientation = 0). We're going to figure out what the threshold is for extended abstracts. 

# In[68]:


merged_df.query("year_float > 2017").words_per_page.hist(bins=40)


# In[69]:


merged_df.query("year_float == 2011 & orientation == 90").words_per_page.hist(bins=20)


# In[70]:


merged_df.query("year_float == 2011 & orientation == 0").words_per_page.hist(bins=20)


# In[71]:


merged_df.query("year_float == 2011 & orientation == 0").sort_values('words_per_page')[['words_per_page']].head(20)


# Extended abstracts are generally less than 500 words/page, but what's going on with these ~400 word/page abstracts? Lets look at the PDF for the one with the fewest words/page:

# In[72]:


WImage(filename="../data/pdfs/2011/p653-jung.pdf")


# Oh no, we can't trust orientation to be a proxy for extended abstract vs paper/note! We'll have to filter by some kind of threshold. 

# In[73]:


merged_df.query("year_float == 2011 & orientation == 90").sort_values('words_per_page')[['words_per_page','num_pages']]


# Before we begin, let's make sure that last extended abstract is actually an EA and not a note in landscape orientation:

# In[74]:


WImage(filename="../data/pdfs/2011/p685-merritt.pdf")


# Whew! Let's go back to that list of papers ordered by words/page for landscape orientation and do some spot checking around the 500-600 words/page break:

# In[75]:


merged_df.query("year_float == 2011 & orientation == 0").sort_values('words_per_page')[['words_per_page']].head(20)


# In[76]:


WImage(filename="../data/pdfs/2011/p709-schwanda.pdf")


# In[77]:


WImage(filename="../data/pdfs/2011/p665-kwon.pdf")


# In[78]:


WImage(filename="../data/pdfs/2011/p151-hu.pdf")


# In[79]:


WImage(filename="../data/pdfs/2011/p359-farnham.pdf")


# OK! Looks like we have a pretty good threshold at 520 words/page: everything below is an extended abstact, everything above is a note.

# In[80]:


len(merged_df)


# In[81]:


merged_df.year_float.value_counts()


# In[82]:


merged_df_all = merged_df


# In[83]:


merged_df = merged_df.query("orientation != 90")


# In[84]:


merged_df = merged_df.query("(year_float > 2017) | (year_float < 2017.5 & words_per_page > 520)")


# In[85]:


counts_before = pd.DataFrame(merged_df_all.year.value_counts()).sort_index()
counts_before.index = counts_before.index.astype(float)


# In[86]:


counts_after = pd.DataFrame(merged_df.year_float.value_counts()).sort_index()


# In[87]:


combi = pd.concat([counts_before,counts_after], axis=1)
combi.columns = ['before','after']
combi


# ## Get title and lead author

# In[88]:


def get_title(text):
    
    head = text[0:300]
    
    if head[0:1] == "\n":
        head = head[1:]

    if head[0:1] == "\n":
        head = head[1:]
    
    
    if len(head.split("\n\n")[0]) > 25:
        base = head.split("\n\n")[0]
    elif len(head.split("\n")[0]) > 25:
        base = head.split("\n")[0]
    elif len(head.split("\n\n")[0] + head.split("\n\n")[1]) > 25:
        base = head.split("\n\n")[0] + head.split("\n\n")[1]
    else:
        base = head.split("\n")[0] + head.split("\n")[1]
        
    title = re.sub("\s+"," ",base.replace("\n", " ").title())
    
    if 'abstract' in title.lower()[0:12]:
        title = title[title.lower().find('abstract')+9:]
        
    if title[0:2] == '[]':
        title = title[2:]
        
    return title


# In[89]:


get_title(merged_df.loc['2010/p117-geiger'].paper_text)


# In[90]:


pd.DataFrame(merged_df['paper_text'].sample(50).apply(get_title))


# In[ ]:





# In[91]:


merged_df['title_from_text'] = pd.DataFrame(merged_df.paper_text.apply(get_title))


# In[93]:


def get_lead_author(filename):
    
    if filename.split('/')[0] != '2016':
        return filename[len(filename.split('-')[0])+1:]
    else:
        return filename[len(filename.split('_')[0])+1:]
        


# In[94]:


merged_df['lead_author'] = merged_df.index.map(get_lead_author)


# In[95]:


merged_df['lead_author'].sample(50)


# ### Does the title have a quotation mark?

# In[96]:


def title_has_quote(title):
    if '“' in title:
        return 1
    elif '”' in title:
        return 1
    elif '"' in title:
        return 1
    elif "''" in title:
        return 1
    else:
        return 0


# In[97]:


merged_df['title_has_quote'] = merged_df['title_from_text'].apply(title_has_quote)
merged_df[['title_has_quote','title_from_text']].sort_values('title_has_quote', ascending=False)


# In[ ]:





# ## Stats for "The Rise and Fall of the Note" in CSCW 2019

# In[98]:


merged_df.query("year_float == 2019").T


# ## Export

# In[99]:


merged_df.query("year_float != 2019").to_csv("../data/cscw-pages-all.csv")


# In[100]:


merged_df.query("year_float != 2019").drop(['ref_text', 'appx_text', 'paper_text'], axis=1).to_csv("../data/cscw-pages-notext.csv")


# In[101]:


end = datetime.datetime.now()

time_to_run = end - start
minutes = int(time_to_run.seconds/60)
seconds = time_to_run.seconds % 60
print("Total runtime: ", minutes, "minutes, ", seconds, "seconds")


# In[ ]:




