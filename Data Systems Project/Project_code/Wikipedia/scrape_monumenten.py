import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents
import csv
from wiki_scraper import process_html_links_into_correct_pages
import pickle

results = []

with open('monumenten_lijsten.csv') as file:
    csvreader = csv.reader(file)
    wiki_lijsten = []
    for row in csvreader:
            wiki_lijsten.append(row[0])

wiki_lijsten = [name.replace(" ", "_") for name in wiki_lijsten]
urls = ["https://nl.wikipedia.org/wiki/" + name for name in wiki_lijsten]

for url in urls: 
    response= requests.get(url)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.findAll("table")
    for table in tables:
        for row in table.find_all('tr'):
            col = row.find('td')
            if col:
                for link in col.find_all('a'):
                    results.append(link['href'])

        break

pages = process_html_links_into_correct_pages(results)

print([page.title for page in pages])
print(len(pages))

with open("monumenten_pages","wb") as file:
    pickle.dump(pages, file)