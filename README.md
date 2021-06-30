
# Amazon Recommendation System - Lab

## Introduction

Now that you've gotten an introduction to collaborative filtering and recommendation systems, it's time to put your skills to test and build a recommendation system for a real world dataset! For this lab, you'll be using a dataset regarding the book reviews on the Amazon marketplace. While the previous lesson focused on user-based recommendation systems, you'll apply a parallel process for an item-based recommendation system to recommend similar books at the bottom of the product page.

## Objectives

In this lab you will: 

- Use graph-based similarity metrics to create a collaborative filtering recommender system

## Load the Dataset


```python
import pandas as pd
import networkx as nx
G = nx.Graph()

df = pd.read_csv('books_data.edgelist', names=['source', 'target', 'weight'], delimiter=' ')
df.head()
```

## Load the Metadata 

Import the metadata available in the file `'books_meta.txt'` (note it is `'\t'` seperated). 


```python
meta = pd.read_csv('books_meta.txt', sep='\t')
meta.head()
```

## Select Books to Test Your Recommender On

Select a small subset of books that you are interested in generating recommendations for. 


```python
# Lets rexamine our fascination with Game of Thrones
GOT = meta[meta.Title.str.contains('Thrones')]
GOT

```

## Generate Recommendations for a Few Books of Choice

The `'books_data.edgelist'` has conveniently already calculated the distance between items for you. Given this preprocessed data, it's time to employ collaborative filtering to generate recommendations! Generate the top 10 recommendations for each book in the subset you chose. Be sure to print the book name that you are generating recommendations for as well as the name of the books being recommended. 


```python
# Well, got a couple or extraneous results in there, but perhaps good measure for comparion.
# What does our recommender return for these books?
rec_dict = {}
id_name_dict = dict(zip(meta.ASIN, meta.Title))
for row in GOT.index:
    book_id = GOT.ASIN[row]
    book_name = id_name_dict[book_id]
    most_similar = df[(df.source == book_id)
                      | (df.target == book_id)
                     ].sort_values(by='weight', ascending=False).head(10)
    most_similar['source_name'] = most_similar['source'].map(id_name_dict)
    most_similar['target_name'] = most_similar['target'].map(id_name_dict)
    recommendations = []
    for row in most_similar.index:
        if most_similar.source[row] == book_id:
            recommendations.append((most_similar.target_name[row], most_similar.weight[row]))
        else:
            recommendations.append((most_similar.source_name[row], most_similar.weight[row]))
    rec_dict[book_name] = recommendations
    print('Recommendations for:', book_name)
    for r in recommendations:
        print(r)
    print('\n\n')
```

## Summary

Well done! In this lab, you effectively created a recommendation system for a real world dataset!
