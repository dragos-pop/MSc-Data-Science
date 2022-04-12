import pandas as pd
import wikipedia
import pickle
from wiki_scraper import check_if_in_amsterdam, find_wiki_pages
import csv

pages = []
titles = []
labels = []

# ADD BEZIENSWAARDIGHEDEN
with open('bezienswaardigheden_pages', "rb") as fp:   # Unpickling
    bezienswaardigheden_pages = pickle.load(fp)
pages += bezienswaardigheden_pages

# ADD monumenten
with open('monumenten_pages', "rb") as fp:   # Unpickling
    monumenten_pages = pickle.load(fp)
pages += monumenten_pages

# make list unique
temp = pages
pages = []
for page in temp:
    if page.title not in titles:
        pages.append(page)
        titles.append(page.title)
        labels.append("Monument of Gebouw")


with open('amsterdamse_onderwerpen', "rb") as fp:   # Unpickling
    ao = pickle.load(fp)


# add amsterdamse onderwerpen at the end of the list with no duplicates and correct labels
current_len = len(pages)
for item in ao:
    page, label, title = item
    adjusted = False
    for i in range(0, current_len):
        if title == titles[i]:
            labels[i] = label
            adjusted = True
            break
    if not adjusted:
        pages.append(page)
        titles.append(title)
        labels.append(label)



with open("page_label_list","wb") as file:
    pickle.dump([pages, labels], file)


print("TOTAL LENGTH NOW ON : " + str(len(pages)))




