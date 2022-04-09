import csv
import os
import sys
import re
import ast
import nltk
import sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from collections import Counter
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from pywsd.utils import lemmatize_sentence
from nltk.probability import FreqDist
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

dir_path = os.path.dirname(os.path.realpath(__file__))
main_data_dir = os.path.join(dir_path, 'TXT')


def plot_sentiment_country_vs_year(country_code):
    """
    This function plots the sentiment scores of a specific country over the years
    :param country_code: the code of the country that you want the sentiment development plot for
    :return:
    """

    plt.figure()
    plt.title('{} speech sentiment'.format(country_code))
    plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
             speeches_df.loc[speeches_df['country'] == country_code]['pos_sentiment'], label='positive', color='green')
    # plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
    #          speeches_df.loc[speeches_df['country'] == country_code]['neu_sentiment'], label='neutral')
    plt.plot(speeches_df.loc[speeches_df['country'] == country_code]['year'],
             speeches_df.loc[speeches_df['country'] == country_code]['neg_sentiment'], label='negative', color='red')
    plt.legend()
    plt.vlines(2011, ymin=0, ymax=0.25, colors='blue', linestyles='--')
    plt.ylim(0, 0.25)
    plt.xlim(1970, 2020)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time (year)', fontsize=16)
    plt.ylabel('Sentiment', fontsize=16)
    plt.savefig('{}_sentiment_development.png'.format(country_code), dpi=300)
    plt.show()

def plot_sentiment_over_year():
    """
    This function plots the sentiment scores of a specific country over the years
    :param country_code: the code of the country that you want the sentiment development plot for
    :return:
    """

    print(speeches_df.groupby('year')['pos_sentiment'].mean().index)
    print(speeches_df.groupby('year')['pos_sentiment'].mean())

    plt.figure()
    plt.title('average speech sentiment')
    plt.plot(speeches_df.groupby('year')['pos_sentiment'].mean().index,
             speeches_df.groupby('year')['pos_sentiment'].mean(), label='positive', color='green')
    # plt.plot(speeches_df.groupby('year')['pos_sentiment'].mean().index,
    #          speeches_df.groupby('year')['neu_sentiment'].mean(), label='neutral')
    plt.plot(speeches_df.groupby('year')['pos_sentiment'].mean().index,
             speeches_df.groupby('year')['neg_sentiment'].mean(), label='negative', color='red')
    plt.legend()
    plt.ylim(0, 0.25)
    plt.xlim(1970, 2020)
    plt.xlabel('Time (year)', fontsize=14)
    plt.ylabel('Sentiment', fontsize=14)
    plt.savefig('average_sentiment_development.png', dpi=300)
    plt.show()


def plot_correlation_matrix(speeches_df, corr_cols):
    """
    This function creates a correlation matrix of specific columns from the speeches df
    :param speeches_df: dataframe containing all the information about the speeches
    :param corr_cols: a list of columns that we cant to calculate the correlation for
    :return:
    """
    matrix = np.triu(speeches_df.loc[:, corr_cols].corr())

    plt.figure()
    sns.heatmap(speeches_df.groupby('country').loc[:, corr_cols].corr(), vmin=-1, vmax=1, center= 0, mask=matrix)
    plt.show()

def sentiment_correlation_countries(speeches_df):

    print(speeches_df['pos_sentiment'])

    pos_sent_df = pd.DataFrame()

    plt.figure()
    sns.heatmap(speeches_df.groupby('country').loc[:, 'pos_sentiment'].corr(), vmin=-1, vmax=1, center= 0)
    plt.show()


def count_most_used_words(data, n):
    """
    This functions a list of the n most used words with their occurrence rate
    :param data: the string to gather the word occurrences from
    :param n: number of most used words in the speeches
    :return: two lists, first with the most occurring words and then their occurrence rate
    """

    occ_words = Counter(data).most_common(n)

    words = [occ_tup[0] for occ_tup in occ_words]
    counts = [occ_tup[1] for occ_tup in occ_words]

    return words, counts


def count_specific_words(data, words):
    """
    This function creates a list of the occurrences of a specific set of words
    :param data: the string to gather the word occurrences from
    :param words: the words to look for in the string
    :return: a list containing the occurrences of the words in order
    """

    occ_words = Counter(data)
    occ_words_in_question = [occ_words[word] for word in words]
    return occ_words_in_question


def count_total_words(data):
    """ This function purely counts the number of words a piece of text

    :param file: a piece of text to count the number of words from
    :return: the word count
    """

    word_count = len(data)

    return word_count


def determine_sentiment(data):
    """
    This function determines the sentiment of a string (in this case a speech)
    :param data: the speech / a string
    :return:
    """

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(data)

    return sentiment


def open_speech(file_path):
    """
    This function opens a file with the correct formatting
    :param file_path:
    :return:
    """

    file = open(file_path, encoding="utf-8-sig")
    data = file.read()

    return data


def top10words_percountry(documents, mode, n=10):
    '''
    Feature vector from unstructured text.
     - Count encoding, used to represent the frequency of words in a vocabulary for a document
     - TF-IDF encoding, used to represent normalized word frequency scores in a vocabulary.
    :param documents: iterable object with speech(es) ['speech1 from Albania', 'speech 2 from Denmark']
    :param mode: string with encoding technique to use. Can be - ['count','tf-idf']
    :return: returns list with top 10 most important words
    '''

    # speech.apply(preprocess_speech())
    if mode == 'count':
        obj = CountVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)
    elif mode == 'tf-idf':
        obj = TfidfVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)

    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    topwords_country = text_features.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n)

    return topwords_country.values.tolist()


def topnwords_peryear(documents, mode, n=10):
    '''
    Vectorizes documents from all years and calculates most important words based on the preferred method.
    :param documents: list with the speeches per year
    :param mode: string with encoding technique to use. Can be - ['count','tf-idf']
    :return: n most important words per year
    '''
    # speech.apply(preprocess_speech())
    if mode == 'count':
        obj = CountVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)
    elif mode == 'tf-idf':
        obj = TfidfVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)
    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    topwords_year = text_features.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n)
    # Get the top 20 words for the whole year
    #avg_tfidf = text_features.mean(axis=0)
    return topwords_year.values.tolist()


def topnwords_perspeech(documents, mode, n=100):

    # speech.apply(preprocess_speech())
    if mode == 'count':
        obj = CountVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)
    elif mode == 'tf-idf':
        obj = TfidfVectorizer(lowercase=True, stop_words='english')
        word_occ = obj.fit_transform(documents)

    text_features = pd.DataFrame(word_occ.toarray(), columns=obj.get_feature_names())
    # Get the top 10 words per country
    top10_country = text_features.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=n)

    return top10_country.values.tolist()

def get_index(mylist, value):
    try:
        index = mylist.index(value)
    except:
        index = np.nan
    return index

def filter_common_words(words):
    common_words = ['united', 'nations', 'country','countries','viet','nam', 'state', 'assembly','today','netherlands',
                    'states','general', 'israeli','people', 'world','peoples', 'region', 'international','syria',
                    'syrian', 'new']
    common_words1 = ['united', 'nation', 'international', 'world', 'country', 'states', 'organization', 'would',
                     'people', 'government', 'general', 'one', 'state', 'time', 'us', 'development']
    return [word for word in words if word not in common_words1]


def preprocess_speech(data):
    """
    This function does the preprocessing
    :param data:
    :return:
    """

    # put all characters in lower case
    data = data.lower()

    # sentences_speech = nltk.sent_tokenize(data)

    # lemmatised_data = []

    data = lemmatize_sentence(data)
        # lemmatised_data.append(lemmas)

    # remove stop words and non-alphabetic stuff from all the text
    sw = nltk.corpus.stopwords.words("english")
    no_sw = []
    for w in data:
        if (w not in sw) and w.isalpha():
            no_sw.append(w)

    return no_sw

def k_means_clustering(data_df, true_k, elbow_plot):
    """
    This functions attempts to find the word groups present in the speeches. Here speeches are considered as samples
    :param data_df: dataframe of all data
    :return:
    """

    data_to_cluster = data_df['speech'].values
    sentences = pd.DataFrame(index=range(8500*200), columns=['sentence'])
    i = 0
    for speech_to_split in tqdm(data_to_cluster, desc='lemmatising speech: '):
        sentences_speech = nltk.sent_tokenize(speech_to_split)
        print(len(sentences_speech))

        for sentence in sentences_speech:
            lemmas = lemmatize_sentence(sentence)
            lemmatized_sentence = ' '.join(word for word in lemmas)
            sentences.iloc[i,0] = lemmatized_sentence

            i += 1

        if i > 30000:
            break

    sentences.dropna(inplace=True)
    print(sentences)
    sentences.to_csv('lemmatized_sentences.csv', sep=';')
    data_to_cluster = sentences['sentence'].values
    # data_to_cluster = [speech_to_split.split('.') for speech_to_split in data_to_cluster]
    # data_to_cluster = sum(data_to_cluster, [])
    # data_to_cluster = np.array(data_to_cluster)
    # print(data_to_cluster)

    # data_to_cluster_tokenised = data_df['speech'].apply(lambda x: nltk.word_tokenize(x))

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data_to_cluster)
    # model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1, verbose=1)
    # model.fit(X)
    #
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    #
    # for i in range(true_k):
    #     print("Cluster % d:" % i),
    #     for ind in order_centroids[i, :10]:
    #         print(' % s' % terms[ind])


    if elbow_plot:
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = np.arange(1, 40, 2)

        for k in tqdm(K, desc='K means elbow plot'):
            model = KMeans(n_clusters=k, init='random', max_iter=300, n_init=1, verbose=1, tol=1)
            model.fit(X)

            distortions.append(sum(np.min(sklearn.metrics.pairwise.pairwise_distances(X, model.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(model.inertia_)

            mapping1[k] = sum(np.min(sklearn.metrics.pairwise.pairwise_distances(X, model.cluster_centers_,
                                           'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = model.inertia_

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.show()


def determine_average_sentence_length(speech_data):
    sentence_lengths = [len(sentence) for sentence in speech_data.split('\n')]

    return np.mean(sentence_lengths)


def remove_line_number(speech):
    '''
    removes the line number at the beginning of speech

    Parameters
    ---------
    speech : str
        piece of text
     '''

    pattern = "\n|^\d+.*?(\w)"
    speech = re.sub(pattern, "\n\g<1>", speech)
    pattern = "\t"
    speech = re.sub(pattern, "", speech)
    pattern = "\n\n"
    speech = re.sub(pattern, "\n", speech)
    pattern = "^\n *"
    speech = re.sub(pattern, "", speech)

    return speech


def happiness_df_cleanup(happiness_df):
    '''
    Creating a dictionary to map the wrong countries name in
    happiness dataframe with correct country name. This is because
    we need to merge happiness dataframe with speeches dataframe
    using country name. And we found cases where country names are
    not the same between these two data set.
    See function check_countryname_consistency as well.
    '''

    country_mapping = {
        'Vietnam' : 'Viet Nam',
        'Moldova' :  'Republic of Moldova',
        'Laos' :  "Lao People's Democratic Republic",
        'Somaliland region' :  'Somalia',
        'Kosovo': None,
        'Taiwan Province of China': None,
        'United Kingdom' :  'United Kingdom of Great Britain and Northern Ireland',
         'United States' :  'United States of America',
        'South Korea' : 'Republic of Korea',
        'Ivory Coast' : 'Côte d’Ivoire',
        'Czech Republic' : 'Czechia',
        'Swaziland' :  'Eswatini',
         'Russia' : 'Russian Federation',
         'Hong Kong S.A.R. of China' : 'China-Hong Kong Special Administrative Region',
         'Palestinian Territories' :  'State of Palestine',
         'Tanzania' :  'United Republic of Tanzania',
         'Syria' : 'Syrian Arab Republic',
         'North Cyprus' :  None,
         'Bolivia' : 'Bolivia (Plurinational State of)',
         'Congo (Kinshasa)' : 'Democratic Republic of the Congo',
        'Venezuela': 'Venezuela (Bolivarian Republic of)',
         'Iran': 'Iran (Islamic Republic of)',
         'Congo (Brazzaville)' :  'Congo'}
    # Replace the country names in happiness dataframe with the correct country names in df-codes.
    happiness_df = happiness_df.reset_index().replace({'Country name': country_mapping} ).set_index(['Country name', 'year'])

    return happiness_df


def speeches_df_cleanup(speeches_df):
    '''
    Replace YDYE (yemen) and POR (Portugal) with correct iso_alpho3 code.
    The remaing 'DDR', 'YUG', 'EU', 'CSK' are not considered countries by
    the UN or don't exist anymore, so we can consider removing them out of
    dataset because we don't have happiness data for these "countries".
    '''
    speeches_df['country'] = speeches_df['country'].str.replace('YDYE','YEM').replace('POR','PRT')
    return speeches_df


def check_countryname_consistency(happiness_df, df_codes):
    happiness_countries = set(happiness_df.reset_index()['Country name'].unique())
    iso_countries = set(df_codes['Country or Area'])
    print(f"The following countries in happiness_df do not appear in iso_countries: {happiness_countries - iso_countries}")


def check_isocode_consistency(speeches_df,df_codes):
    speech_codes = set(speeches_df['country'].unique())
    iso_codes = set(df_codes['ISO-alpha3 Code'].unique())
    print(f"The following codes in speeches_df do not appear in df_codes: {speech_codes - iso_codes}")


def interpolate(df, col, country):
    minidf = pd.DataFrame({col : df[col].loc[:,country].interpolate(method='slinear')})
    minidf['country'] = country
    minidf = minidf.reset_index()
    minidf.set_index(['year', 'country'], inplace=True)
    return df.update(minidf)


def multi_interpolate(df, columns, countries):
    for country in tqdm(countries):
        for column in columns:
            try:
                interpolate(df, column, country)
            except:
                print(f"FAIL: {country} : {column}")
    return df


def preprocess_speech_(speech):
    """ This function tokenizes (splits in words and eliminate
    punctuation) and lowercases the speech

    :param file: string
    :return: list of lowercase words
    """
    words = nltk.word_tokenize(speech)
    r = []
    for w in words:
        r.append(w.lower())
    return r


def count_list_occurences(list, x):
    """ This function counts the number of times an element appears
    in a list

    :param file: a list and an element
    :return: the number of times the element appeared in list
    """
    counter = 0
    for i in list:
        if x == i:
            counter += 1
    return counter


def count_referenced_countries(data, countries, years):
    """ This function iterates over the speeches in the selected years
    and counts the number of times each country appeared in each speech,
    and the number of speeches that mentioned the respective country

    :param file: the dataframe, a list of countries and years
    :return: the number of references of each country per year and
    the number of speeches referencing the respective country per year
    """
    for y in years:
        year = data.loc[(y)]
        year['processed_speech'] = year.apply(lambda row: preprocess_speech_(row['Speech']), axis=1)
        for i in countries:
            s = str(i) + "_ref_count_" + str(y)
            year[s] = year.apply(lambda row: count_list_occurences(row['processed_speech'], i), axis=1) #only looks for usa, not united states, america etc
            print("# references", i, '(', y, ')', sum(year[s]))
            s1 = "bool_" + str(i) + "_ref_" + str(y)
            year[s1] = np.where(year[s]== 0, False, True)
            print("# speeches referencing", i, '(', y, ')', sum(year[s1]))

def generate_wordcloud(important_words, year):
    '''
    Generate word cloud from list of most important words. Plots it.
    :param important_words: list with the top words.
    :return: image
    '''

    word_counter = Counter(important_words)
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(word_counter)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(str(year))
    plt.show()

    return wordcloud

if __name__ == '__main__':

    # True --> run preprocessing and save the results, False --> just do the data analysis with your previously saved
    # dataframe file (always have to do a preprocessing run to save the dataframe of course)
    do_preprocessing = False

    if do_preprocessing:
        speeches_df = pd.DataFrame(columns=['session_nr', 'year', 'country', 'word_count', 'pos_sentiment',
                                            'neu_sentiment', 'neg_sentiment', 'average_sentence_length',
                                            'most_used_words', 'speech', 'preprocessed_speech'])

        num_directories = len(next(os.walk(main_data_dir))[1])

        # loop through all directories of the data
        for root, subdirectories, files in tqdm(os.walk(main_data_dir), total=num_directories, desc='directory: '):

            # remove all the files starting with '.' (files created by opening a mac directory on a windows PC,
            # so will only do something if you are working on a windows PC
            files_without_dot = [file for file in files if not file.startswith('.')]

            # loop through files and extract data
            for file in tqdm(files_without_dot, desc='files: ', leave=False):
                country, session_nr, year = file.replace('.txt', '').split('_')

                # open a speech with the correct formatting
                speech_data = open_speech(os.path.join(root, file))
                speech_data = remove_line_number(speech_data)

                # preprocess the data
                preprocessed_bag_of_words = preprocess_speech(speech_data)

                # calculate all the features through functions
                word_count = count_total_words(preprocessed_bag_of_words)
                most_used_words = count_most_used_words(preprocessed_bag_of_words, 20)
                occs_of_spec_words = count_specific_words(preprocessed_bag_of_words, ['economy'])
                sentiment_of_speech = determine_sentiment(speech_data)
                average_sentence_length = determine_average_sentence_length(speech_data)

                # append the line of features to the dataframe
                speeches_df = speeches_df.append({'session_nr': int(session_nr),
                                                  'year': int(year),
                                                  'country': country,
                                                  'word_count': word_count,
                                                  'pos_sentiment': sentiment_of_speech['pos'],
                                                  'neu_sentiment': sentiment_of_speech['neu'],
                                                  'neg_sentiment': sentiment_of_speech['neg'],
                                                  'average_sentence_length': average_sentence_length,
                                                  'most_used_words': most_used_words,
                                                  'speech': speech_data,
                                                  'preprocessed_speech': preprocessed_bag_of_words
                                                  },
                                                 ignore_index=True)



        # read in country codes and happiness data
        df_codes = pd.read_csv('UNSD — Methodology.csv', delimiter=',')
        happiness_df = pd.read_excel('DataPanelWHR2021C2.xls', index_col=[0, 1])

        # check if country names and iso codes are consistent before merging
        check_isocode_consistency(speeches_df, df_codes)
        check_countryname_consistency(happiness_df, df_codes)

        # cleanup dataframes before merging
        speeches_df = speeches_df_cleanup(speeches_df)
        happiness_df = happiness_df_cleanup(happiness_df)

        speeches_df = speeches_df.merge(df_codes, how='left', left_on='country', right_on='ISO-alpha3 Code')

        # add the happiness dataframe to the

        speech_happi_merged_df = pd.merge(speeches_df, happiness_df, how='left',
                                          left_on=['year', 'Country or Area'], right_on=['year', 'Country name'])

        speech_happi_merged_df.to_csv('preprocessed_dataframe.csv')

    # get into the exploration after reading the saved file
    speeches_df = pd.read_csv('preprocessed_dataframe.csv')
    # Turn preprocessed speech into text
    speeches_df['preprocessed_speech'] = speeches_df['preprocessed_speech'].apply(lambda x: " ".join(filter_common_words(ast.literal_eval(x))))


    # # Create the Empty Lists
    # top10_aux = []
    # texts_over_years = []
    # # Loop through the years
    # for year in range(speeches_df['year'].min(), speeches_df['year'].max()+1):
    #     # Select view of the year
    #     year_data = speeches_df[speeches_df['year'] == year]
    #     # Calculate top 10 per year
    #     topwords_countryyear = top10words_percountry(year_data['preprocessed_speech'], mode='tf-idf')
    #     # Append to Dataframe
    #     top10_aux += topwords_countryyear
    #     # Aggregate all speeches
    #     year_text = ' '.join(speeches_df[speeches_df['year'] == year]['preprocessed_speech'])
    #     # Append Dataframe
    #     texts_over_years.append(year_text)
    #
    # # Add column 'top10words' to main df for each speech
    # speeches_df['top10words'] = top10_aux
    # # Get main words throughout all years
    # words_over_years_df = pd.DataFrame({'years': [x for x in range(1970, 2021)],
    #                                     'topwords': topnwords_peryear(texts_over_years, mode='tf-idf', n=20)})
    #
    # speeches_df.to_csv('preprocessed_dataframe_top10.csv')
    # speeches_df.to_pickle('preprocessed_dataframe_top10.pkl')
    #
    # #Generate Wordclouds for different years
    # for year in range(2016,2021):
    #     year_words = words_over_years_df[words_over_years_df.years == year]['topwords'].values
    #     generate_wordcloud(year_words[0], year)


    # do k_means clustering
    true_k = 8
    elbow_plot = True
    # k_means_clustering(speeches_df, true_k, elbow_plot)
    # number of covid and covid synonyms mentions in 2020 speeches
    # x = df.loc[(2020)]
    # x['processed_speech'] = x.apply(lambda row: preprocess_speech_(row['Speech']), axis=1)
    # x['covid'] = x.apply(lambda row: count_list_occurences(row['processed_speech'], 'covid-19')+count_list_occurences(row['processed_speech'], 'corona')+count_list_occurences(row['processed_speech'], 'coronavirus')+count_list_occurences(row['processed_speech'], 'sars-cov-2'), axis=1)
    # print("# of times coronavirus was mentioned in China's 2020 speech:", x['covid'].loc['CHN'])

    # number of exclamation marks
    # df['exclamation_marks'] = df.apply(lambda row: row['Speech'].count('!'), axis=1)
    # print("# of exclamations marks in China's 2020 speech:", df['exclamation_marks'].loc[(2020, 'CHN')])

    # countries_ = ['usa', 'china']
    # years_ = [2017, 2018, 2019, 2020]
    # count_referenced_countries(df, countries_, years_) #only looks for usa, not united states, america etc; they can be added to countries_ and then summed tohether


    # corr_cols = ['year', 'word_count', 'pos_sentiment', 'neg_sentiment', 'neu_sentiment',
    #              'average_sentence_length', "Life Ladder", "Log GDP per capita", 'Social support', 'Freedom to make '
    #              'life choices', 'Generosity', 'Perceptions of corruption']
    # plot_correlation_matrix(speeches_df, corr_cols)
    # sentiment_correlation_countries(speeches_df)

    # plot some figures from the data
    plot_sentiment_country_vs_year('SYR')
    # plot_sentiment_country_vs_year('USA')

    plot_sentiment_over_year()


    # interpolation for missing values in happiness df columns

    countries = speeches_df.reset_index()['country'].unique()
    cols = ['Life Ladder', 'Log GDP per capita', 'Social support',
            'Healthy life expectancy at birth', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption', 'Positive affect',
            'Negative affect']

    speeches_df.set_index(['year', 'country'], inplace=True)

    #plot before interpolating
    fig, axs = plt.subplots()
    speeches_df['Log GDP per capita'].loc[:,'CYP'].plot(ax=axs, style = '.')
    axs.set_ylabel('Log GDP per capita')
    axs.set_title("Cyprus")
    plt.show()

    multi_interpolate(speeches_df, cols, countries)

    #plot after interpolating
    fig, axs = plt.subplots()
    speeches_df['Log GDP per capita'].loc[:,'CYP'].plot(ax=axs, style = '.')
    axs.set_ylabel('Log GDP per capita')
    axs.set_title("Cyprus")
    plt.show()

    #plotting how most common words evolve over years per country

    fig, axs = plt.subplots(2,2,  sharey=True, figsize = (13,13))
    for i, country in enumerate(['SYR', 'USA', 'NLD','VNM']):
    #     try:
            ax = axs.flatten()[i]
            times = np.arange(2010,2021)
            top_per_year_country= topnwords_perspeech([speeches_df[(speeches_df['year']==year) & (speeches_df['country'] == country)]['speech'].values[-1] for year in times], 'tf-idf', n=100)
            #take all speeches of a specific country
            merged_speeches = " ".join(speeches_df[speeches_df['country'] == country]['preprocessed_speech'].values)
            top_allyears = topnwords_perspeech([merged_speeches], 'tf-idf', n=100)
            top_allyears = [filter_common_words(top_allyears[0])]
            score_dict = {}
            for word in top_allyears[0][:8]:
                indices = [100-get_index(top_per_year_country[i],word) for i in range(len(top_per_year_country))]
                score_dict[word] = indices
            for word in score_dict.keys():
    #             None
                size = [0  if np.isnan(n) else n for n in score_dict[word]]
                ax.scatter(times, score_dict[word], label = word, s=size)
            ax.legend()
            ax.set_xlabel("Year")
            ax.set_ylabel("Score")
            ax.set_title(country)
    #     except:
            pass
    plt.show()
