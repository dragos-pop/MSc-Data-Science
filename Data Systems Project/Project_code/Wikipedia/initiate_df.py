import pandas as pd
import pickle
import re

def remove_brackets(wiki_title):
    return re.sub(r"\([^()]*\)", "", wiki_title)

with open('page_label_list', "rb") as fp:   # Unpickling
    page_list, label_list = pickle.load(fp)

name_list = [remove_brackets(page.title) for page in page_list]

url_list = [page.url for page in page_list]

d = {'name': name_list, 'label': label_list, 'wiki_page_obj': page_list, 'wiki_url' : url_list}
df = pd.DataFrame(data=d)
df.to_csv("Cultural_features_final.csv", index= False)
