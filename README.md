# The Rise and Fall of the Note: Changing Paper Lengths in ACM CSCW, 2000-2018

By R. Stuart Geiger, staff ethnographer, Berkeley Institute for Data Science, UC-Berkeley

This repo contains the code and data needed to reproduce the figures in [a forthcoming paper](paper/camera-ready-paper.pdf) in _Proceedings of the ACM on Human-Computer Interaction_ -- the new journal venue for the proceedings of the ACM conference on Computer-Supported Cooperative Work (or CSCW). The entire study involved text analysis of copyrighted papers, which is not free to redistribute here. However, the notebook I used for processing the PDFs is available for reference at `data-processing.ipynb`. A data file containing all the quantitative statistics for each paper is at `cscw-pages-notext.csv`. This file is loaded by `analysis-viz.ipynb`, which processes it to produce the statistics and graphs presented in the paper. This notebook can also be run interactively for free in the cloud with Binder, so you can change various parameters or visualize it differently.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/staeiou/cscw19-paper-lengths/master?filepath=analysis-viz.ipynb)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3380345.svg)](https://doi.org/10.5281/zenodo.3380345)


## Abstract

In this note, I quantitatively examines various trends in the lengths of published papers in ACM CSCW from 2000-2018, focusing on several major transitions in editorial and reviewing policy. The focus is on the rise and fall of the 4-page note, which was introduced in 2004 as a separate submission type to the 10-page double-column "full paper" format. From 2004-2012, 4-page notes of 2,500 to 4,000 words consistently represented about 20-30% of all publications. In 2013, minimum and maximum page lengths were officially removed, with no formal distinction made between full papers and notes. The note soon completely disappeared as a distinct genre, which co-occurred with a trend in steadily rising paper lengths. I discuss such findings both as they directly relate to local concerns in CSCW and in the context of longstanding theoretical discussions around genre theory and how socio-technical structures and affordances impact participation in distributed, computer-mediated organizations and user-generated content platforms. There are many possible explanations for the decline of the note and the emergence of longer and longer papers, which I identify for future work. I conclude by addressing the implications of such findings for the CSCW community, particularly given how genre norms impact what kinds of scholarship and scholars thrive in CSCW, as well as whether new top-down rules or bottom-up guidelines ought to be developed around paper lengths and different kinds of contributions.
