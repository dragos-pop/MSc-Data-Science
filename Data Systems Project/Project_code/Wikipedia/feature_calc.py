from asyncio.windows_events import NULL
from dataclasses import replace
import wikipedia
import requests
wikipedia.set_lang("nl")  
from bs4 import BeautifulSoup as bs 

#set correct user agent header
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36"}

# this function counts the nr. of vieuws of a given wikipedia page in the period start to end "YYYYMMDD"
def count_total_views(page, start, end, lang):
    total_views = 0
    if page.title:
        url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/" + lang + ".wikipedia/all-access/all-agents/" + page.title.replace(" ", "_") + "/monthly/" + start + "/" + end
        try:
            timeseries = requests.get(url, headers = headers).json()["items"]
            for t in timeseries:
                total_views += t["views"]
            return total_views
        except:
            return None

def get_coordinates(page):
    URL = "https://nl.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": page.title,
        "prop": "coordinates"
    }

    R = requests.get(url=URL, params=PARAMS)
    DATA = R.json()
    PAGES = DATA['query']['pages']

    for k, v in PAGES.items():
        try:
           return (v['coordinates'][0]['lat'], v['coordinates'][0]['lon'] )
        except:
            return None



# this function counts the nr of views on the english wikipedia page
def count_english_views(page, start, end):
    wikipedia.set_lang('en')
    try:
        page = wikipedia.page(page.title)
        return count_total_views(page, start, end, 'en')
    except:
        return 0
        

# this function counts nr number of languages in wich the page has a translation
def count_nr_of_languages(page):

    URL = "https://nl.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "titles": page.title,
        "prop": "langlinks",
        "format": "json"
    }

    R = requests.get(url=URL, params=PARAMS)

    DATA = R.json()['query']['pages']
    if 'langlinks' in DATA[list(DATA.keys())[0]].keys():
        count = len(DATA[list(DATA.keys())[0]]['langlinks']) + 1
    else:
        count = 1

    return count
    


# this function returns the number of edits that have been done to the dutch version of the page
def count_nr_of_edits(page):
    BASE_URL = "http://nl.wikipedia.org/w/api.php"
    TITLE = page.title

    parameters = { 'action': 'query',
            'format': 'json',
            'continue': '',
            'titles': TITLE,
            'prop': 'revisions',
            'rvprop': 'ids|userid',
            'rvlimit': 'max'}

    wp_call = requests.get(BASE_URL, params=parameters)
    response = wp_call.json()

    total_revisions = 0

    while True:
        wp_call = requests.get(BASE_URL, params=parameters)
        response = wp_call.json()

        for page_id in response['query']['pages']:
            total_revisions += len(response['query']['pages'][page_id]['revisions'])

        if 'continue' in response:
            parameters['continue'] = response['continue']['continue']
            parameters['rvcontinue'] = response['continue']['rvcontinue']

        else:
            break

    return total_revisions


# returns the amount of characters in a given wikipedia page.
def get_page_length(page):
    if page:
        return len(page.content)
    else:
        return 0



