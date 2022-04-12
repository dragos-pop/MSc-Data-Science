import wikipedia
wikipedia.set_lang('nl')

# this function gets in a list of wiki link names and returns the actual pages. 
def find_wiki_pages(titles):
    pages = []
    for link in titles:
        print(str(len(pages)) + " out of " + str(len(titles)))
        try:
            page = wikipedia.page(link)
            pages.append(page)
        except wikipedia.exceptions.DisambiguationError as e:
            for option in e.options:
                if "Amsterdam" in option:
                    try: 
                        page = wikipedia.page(option)
                        pages.append(page)
                    except wikipedia.exceptions.DisambiguationError:
                        print("DisambiguationError: " + link)
                    except wikipedia.exceptions.PageError:
                        print("PageError: " + link)
                else: 
                    print("DisambiguationError: " + link)
        except wikipedia.exceptions.PageError:
            print("PageError: " + link)
    
    return pages

# this function checks if the word "Amsterdam" or "Amsterdamse" occurs in the first 300 charakters of the page's summary.
def check_if_in_amsterdam(summary):

    first_part = summary[:300]

    if "Amsterdam" in first_part or "Amsterdamse" in first_part:
        return True
    else:
        return False

# this functions transforms the parser output links into wiki pages. 
def process_html_links_into_correct_pages(links):
    # order randomly and with unique values
    # also, reformat link name
    links = list(set([link.replace("/wiki/", '').replace("_", " ") for link in links]))

    # find appropriate wiki page
    pages = find_wiki_pages(links)

    # do the in amsterdam check
    pages = [page for page in pages if check_if_in_amsterdam(page.summary)]

    return pages