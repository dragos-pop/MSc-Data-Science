import csv
import pickle
from turtle import title
from feature_calc import count_english_views, count_nr_of_languages, count_total_views, count_nr_of_edits, get_coordinates
import pandas as pd

#this function takes in a list and outputs the min max feature scaling of that list
def norm(a):
    n = []
    amin, amax = min([i for i in a if i != None]), max([i for i in a if i != None])
    for _, val in enumerate(a):
        if val != None:
            n.append((val-amin) / (amax-amin))
        else:  
            n.append(None)
    return n


with open('page_label_list', 'rb') as fp:
    pages, _ = pickle.load(fp)


df = pd.read_csv ('Cultural_features_final.csv')
views = []
en_views = []
edits = []
languages = []
coordinates = []

for page in pages:
    print(str(len(coordinates)) + ' out of ' + str(len(pages)))
    # views.append(count_total_views(page, "20020101", "20200101", "nl"))
    # en_views.append(count_english_views(page, "20020101", "20200101"))
    # edits.append(count_nr_of_edits(page))
    # languages.append(count_nr_of_languages(page))
    coordinates.append(get_coordinates(page))


print(coordinates)

# df['wiki_nl_views'] = views
# df['wiki_norm_nl_views'] = norm(views)
# df['wiki_eng_views'] = en_views
# df['wiki_norm_eng_views'] = norm(en_views)
# df['wiki_nr_of_edits'] = edits
# df['wiki_norm_nr_of_edits'] = norm(edits)
# df['wiki_nr_of_languages'] = languages
# df['wiki_norm_nr_of_languages'] = norm(languages)
df['wiki_coordinates'] = coordinates

df.to_csv("Cultural_features_final.csv", index=False)

print(df)




