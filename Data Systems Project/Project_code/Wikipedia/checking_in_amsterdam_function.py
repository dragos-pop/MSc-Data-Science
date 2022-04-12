import csv
from wiki_scraper import find_wiki_pages, check_if_in_amsterdam

page_list = []

# ADD BEZIENSWAARDIGHEDEN
title_list = []
with open('bezienswaardigheden.csv') as file:
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
            rows.append(row[0])
    title_list += rows


page_list += find_wiki_pages(title_list)

nr_of_pages_in_amsterdam = len(page_list)

nr_of_verified_pages = sum([check_if_in_amsterdam(page.summary) for page in page_list])

print("number of pages in Amsterdam: " + str(nr_of_pages_in_amsterdam))
print("of those " + str(nr_of_pages_in_amsterdam) + " pages, " + str(nr_of_verified_pages) + " were verified correctly")