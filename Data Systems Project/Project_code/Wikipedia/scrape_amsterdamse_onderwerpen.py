import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
from wiki_scraper import process_html_links_into_correct_pages
import pickle

pages_labels_and_titles = []
response= requests.get("https://nl.wikipedia.org/wiki/Lijsten_van_Amsterdamse_onderwerpen")
soup = BeautifulSoup(response.text, 'html.parser')

# this function turs a list from the Amsterdamse onderwerpen page, into wikipages with labels and titles. 
def add_wiki_list_to_pages_labels_and_titles(list, label):
    #find wiki url 
    html_links = []
    for item in list:
        a = item.find('a')
        if not isinstance(a, int) and not isinstance(a, type(None)):
            html_links.append(a['href'])
    pages = process_html_links_into_correct_pages(html_links)
    
    # add to list without duplicates
    existing_titles = [title for _,_,title in pages_labels_and_titles]
    for page in pages:
        if page.title not in existing_titles:
            pages_labels_and_titles.append([page, label, page.title])


lists = soup.find_all('ul')

##code used to select correct lists
# counter = 0
# for item in lists:
#     print("nr. : " + str(counter))
#     print(item.find("li"))
#     counter +=1

# process all categories on the Amsterdamse onderwerpen page. 
add_wiki_list_to_pages_labels_and_titles(lists[10], "Monument of Gebouw")
add_wiki_list_to_pages_labels_and_titles(lists[11], "Monument of Gebouw")
add_wiki_list_to_pages_labels_and_titles(lists[13], "Herdenkings Monument")
add_wiki_list_to_pages_labels_and_titles(lists[14], "Herdenkings Monument")
add_wiki_list_to_pages_labels_and_titles(lists[15], "Museum")
add_wiki_list_to_pages_labels_and_titles(lists[17], "Concertzaal of Theater")
add_wiki_list_to_pages_labels_and_titles(lists[18], "Concertzaal of Theater")
add_wiki_list_to_pages_labels_and_titles(lists[20], "Bioscoop")
add_wiki_list_to_pages_labels_and_titles(lists[21], "Bioscoop")
add_wiki_list_to_pages_labels_and_titles(lists[24], "Park of recreatiegebied")
add_wiki_list_to_pages_labels_and_titles(lists[25], "Park of recreatiegebied")
add_wiki_list_to_pages_labels_and_titles(lists[27], "Markt")
add_wiki_list_to_pages_labels_and_titles(lists[28], "Markt")


# save
with open('amsterdamse_onderwerpen', "wb") as fp:   # Unpickling
    pickle.dump(pages_labels_and_titles, fp)
