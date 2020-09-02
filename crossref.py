import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from habanero import Crossref



from PIL import Image

#set title

image = Image.open('image/crossreflogo.png')
st.image(image, width = 800)

def main():
    activities = ['What is CrossRef API', 'CrossRef Exploration',
    'Do Your Own Simple Query']
    option = st.sidebar.selectbox('Selection Option:', activities)

#Intro
    if option == 'What is CrossRef API':
        st.title("Crossref API Intro")
        crossref_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#313F3D;text-align:center;">Crossref API</h3>
        </div>
        """
        st.markdown(crossref_temp,unsafe_allow_html=True)

        crossref_write = """
        Crossref is an API that allows researchers to find links to research much easier.

            * Providing
                * Digital Object Identifier
                * Title
                * URLS
                * The page number
                * Publisher
                * And much more!

        For example if I want to search for articles on dogs.

        ```python
        dog_search = cr.types(ids = "journal-article", works = True, query = 'dog',
        cursor = "*", offset=None, cursor_max=10000, limit = 1000, progress_bar = True,
               filter = {'from-created-date':'2020-01-01', 'until-created-date':'2020-09-01'})
        ```

        * Here we use cross reference types to look for a certain type of works. Mainly Journal-Articles.
        * The query is dog
        * cursor_max allows us to increase the search of the limit which is 1000 allowing us to do a deep page through
        by setting cursor = "*"

        * We also can filter by date.

            * There are different date types we can choose.
                * from-index-date
                * from-update-date
                * from-deposit-date
                * from-created-date
                * from-pub-date

        * When using time filters to retrieve periodic, incremental metadata updates, the from-index-date filter


        * select can also be used here for example select = ['DOI', 'title', 'publisher']

        To access the content you want you would run a list comprehension.

        ```python
        items = [z['message']['items'] for z in dog_search]
        items = [item for sublist in items for item in sublist]
        ```

        This would return a list that contains dictionaries which you can access the information for. This is looking at items[0].keys()

        ```python
        dict_items([('indexed', {'date-parts': [[2020, 3, 24]], 'date-time': '2020-03-24T17:41:15Z', 'timestamp': 1585071675561}), ('update-to', [{'updated': {'date-parts': [[2020, 3, 24]], 'date-time': '2020-03-24T00:00:00Z', 'timestamp': 1585008000000}, 'DOI': '10.1371/journal.pone.0227253', 'type': 'correction', 'label': 'Correction'}]), ('reference-count', 1), ('publisher', 'Public Library of Science (PLoS)'), ('issue', '3'), ('license', [{'URL': 'http://creativecommons.org/licenses/by/4.0/', 'start': {'date-parts': [[2020, 3, 24]], 'date-time': '2020-03-24T00:00:00Z', 'timestamp': 1585008000000}, 'delay-in-days': 0, 'content-version': 'vor'}]), ('content-domain', {'domain': ['www.plosone.org'], 'crossmark-restriction': False}), ('short-container-title', ['PLoS ONE']), ('DOI', '10.1371/journal.pone.0231043'), ('type', 'journal-article'), ('created', {'date-parts': [[2020, 3, 24]], 'date-time': '2020-03-24T17:26:43Z', 'timestamp': 1585070803000}), ('page', 'e0231043'), ('update-policy', 'http://dx.doi.org/10.1371/journal.pone.corrections_policy'), ('source', 'Crossref'), ('is-referenced-by-count', 0), ('title', ['Correction: Assertive, trainable and older dogs are perceived as more dominant in multi-dog households']), ('prefix', '10.1371'), ('volume', '15'), ('author', [{'given': 'Lisa J.', 'family': 'Wallis', 'sequence': 'first', 'affiliation': []}, {'given': 'Ivaylo B.', 'family': 'Iotchev', 'sequence': 'additional', 'affiliation': []}, {'given': 'Enikő', 'family': 'Kubinyi', 'sequence': 'additional', 'affiliation': []}]), ('member', '340'), ('published-online', {'date-parts': [[2020, 3, 24]]}), ('reference', [{'issue': '1', 'key': 'pone.0231043.ref001', 'doi-asserted-by': 'crossref', 'first-page': 'e0227253', 'DOI': '10.1371/journal.pone.0227253', 'article-title': 'Assertive, trainable and older dogs are perceived as more dominant in multi-dog households', 'volume': '15', 'author': 'LJ Wallis', 'year': '2020', 'journal-title': 'PLoS ONE'}]), ('container-title', ['PLOS ONE']), ('language', 'en'), ('link', [{'URL': 'https://dx.plos.org/10.1371/journal.pone.0231043', 'content-type': 'unspecified', 'content-version': 'vor', 'intended-application': 'similarity-checking'}]), ('deposited', {'date-parts': [[2020, 3, 24]], 'date-time': '2020-03-24T17:26:47Z', 'timestamp': 1585070807000}), ('score', 7.0901704), ('issued', {'date-parts': [[2020, 3, 24]]}), ('references-count', 1), ('journal-issue', {'published-online': {'date-parts': [[2020, 3, 24]]}, 'issue': '3'}), ('URL', 'http://dx.doi.org/10.1371/journal.pone.0231043'), ('relation', {'cites': []}), ('ISSN', ['1932-6203']), ('issn-type', [{'value': '1932-6203', 'type': 'electronic'}]), ('subject', ['General Biochemistry, Genetics and Molecular Biology', 'General Agricultural and Biological Sciences', 'General Medicine'])])
        ```

        In the exploration section we will flesh this out.
        """

        st.markdown(crossref_write,unsafe_allow_html=True)


#Exploration
    elif option == 'CrossRef Exploration':
        st.title("Exploring the Animal Query")
        html_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Animal Queries</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


        write_up = """
        I used the python version of crossref using hanabero.

        ```python
        #load in libraries for wrangling
        from habanero import Crossref
        import pandas as pd
        import numpy as np
        cr = Crossref()
        ```
        I decided to go with cr.types instead of cr.works because I wanted specific types of write ups.
        Which were articles or journals and not books.

        To gather the information I broke it up into two queries.  This was because of the limit of 10000 even with cursor.

        ```python
        #Split up into two queries from May 1st to September 1st 2020
        res = cr.types(ids = "journal-article", works = True, query = 'animal',  cursor = "*",
               offset=None, cursor_max=10000, limit = 1000, progress_bar = True,
               filter = {'from-created-date':'2020-05-01', 'until-created-date':'2020-09-01'})

        #First half of the year January 1st, 2020 to April 30th, 2020
        res2 = cr.types(ids = "journal-article", works = True, query = 'animal',  cursor = "*",
               offset=None, cursor_max=10000, limit = 1000, progress_bar = True,
               filter = {'from-created-date':'2020-01-01', 'until-created-date':'2020-04-30'})

        ```

        I then accessed the nested dictionary information with a list comprehension.

        ```python
        #Collect article information within the nested dictionary
        items = [z['message']['items'] for z in res]
        items = [item for sublist in items for item in sublist]

        items2 = [z['message']['items'] for z in res2]
        items2 = [item for sublist in items2 for item in sublist]
        ```

        I also decided to explore the keys to look for any categories that might be interesting. (Yes I love the number 7.)

        ```python
        #Look through interesting features to look for
        print(items[7].keys())
        print()
        print(items[77].keys())
        print()
        print(items[777].keys())
        print()
        ```
        Which returned this
        """
        st.markdown(write_up, unsafe_allow_html=True)

        image = Image.open('image/dictkeys.png')
        st.image(image, width = 800)


        write_up2 = """
        I chose these categories and created lists to append them later with a helper fucntion.

        ```python
        #Empty List to collect the features of interest
        publisher = []
        title = []
        DOI = []
        created = []
        url = []
        score = []
        page = []
        issn = []
        subject = []
        author_first = []
        author_last = []
        container_title = []
        language = []

        #Helper function to do feature extraction
        def extract(data):
            for x in range(0, len(data)):
            publisher.append(data[x]['publisher'])
            title.append(data[x]['title'])
            DOI.append(data[x]['DOI'])
            created.append(data[x]['created']['date-time'])
            url.append(data[x]['URL'])
            score.append(data[x]['score'])
            container_title.append(data[x]['container-title'])
            try:
                language.append(data[x]['language'])
            except:
                language.append('none')
            try:
                page.append(data[x]['page'])
            except:
                page.append('none')

            try:
                subject.append(data[x]['subject'])
            except:
                subject.append('none')
            try:
                issn.append(data[x]['ISSN'])
            except:
                issn.append('none')

            try:
                author_first.append(list(map(lambda x: x["given"], data[x]['author'])))
            except:
                author_first.append('none')
            try:
                author_last.append(list(map(lambda x: x["family"], data[x]['author'])))
            except:
                author_last.append('none')

        ```

        I then put everything in a dataframe to make the next step of cleaning easier.

        ```python
        #Create DateFrame to view easier
        df = pd.DataFrame()
        df['publisher'] = publisher
        df['title'] = title
        df['DOI'] = DOI
        #change date to datetime for better organization
        df['created'] = created
        df['created'] = pd.to_datetime(df['created'].astype(str), format="%Y-%m-%d %H:%M:%S.%f")
        df['url'] = url
        df['score'] = score
        df['page'] = page
        df['ISSN'] = issn
        df['subject'] = subject
        df['author_first'] = author_first
        df['author_last'] = author_last
        df['lang'] = language
        df['container_title'] = container_title

        ```

        These were some of the Cleaning Tasks I performed.

            * publisher nothing to change
            * title get rid of list brackets and do regex to get rid of \\r\\n which probably means the next line, there are some titles that need to remove emails as well as filling in missing titles and translating titles to english
            * DOI Doesn't look like anything needs to be changed
            * created nothing to change set as index
            * url nothing to change might be interesting to use the summarizer api to pull content from there
            * score nothing to change
            * page different formats of paging number will have to see how we replace the nones maybe do nothing about it
            * ISSN pull out of list make into str
            * subject pull out of list
            * author will have to pull out of list and combine the first and last name and pull them in a string format in one column

        First step was to set the 'nones' from the try and except to empty strings.

        ```python
        #Set 'none' to empty strings
        col_str = ['title', 'author_first', 'author_last', 'ISSN', 'subject', 'lang']

        for col in col_str:
            df2[col] = df2[col].replace('none', '')
        ```

        Also combine the authors first and last name

        ```python

        #combine first and last name through zip
        df2['authors'] = df2.apply(lambda x: [m + ' ' + n for m,n in zip(x['author_first'], x['author_last'])], 1)
        ```

        Then changed the columns that were lists to strings for NLP later on.

        ```python
        #change list to strings
        col_str = ['title', 'authors', 'ISSN', 'subject', 'container_title']

        for col in col_str:
            df2[col] = [','.join(map(str, l)) for l in df2[col]]
        ```
        Got rid of the +00:00 for better looks of the dataframe

        ```python
        #remove those annoying +00:00 from datetime and set it as index
        import datetime

        #change back to string and do index slicing to remove the 0's and then change back to datetime
        df2['created'] = df2['created'].apply(lambda x: str(x))
        df2['created'] = df2['created'].apply(lambda x: x[:-6])
        df2['created'] = pd.to_datetime(df2['created'].astype(str), format="%Y-%m-%d %H:%M:%S")
        df2 = df2.rename(columns = {'created': 'date'})
        ```
        Looked at the titles to make sure to remove as some of them were not English and had extra languages, I also removed emails from titles.

        ```python
        #Remove all other languages but English from the titles
        import string
        #choose only english
        printable = set(string.printable)

        #apply to our title
        df2['title'] = df2['title'].apply(lambda y: ''.join(filter(lambda x: x in printable, y)))


        #Removing Emails in title
        import re

        email = '\S*@\S*\s?'
        pattern = re.compile(email)

        #removing emails using regex
        df2['title'] = df2['title'].apply(lambda x: pattern.sub('', x))
        ```

        To put back the titles that were now gone because they weren't English I used newspaper3k API to bring them back.

        ```python
        #use the newspaper3k API
        import nltk
        from newspaper import Article

        nltk.download('punkt')


        #For loop for getting the title out
        article_title = []

        for idx in no_title_index:
            url = no_title['url'][idx]
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            article_title.append(article.title)

        #set the title to what was extracted from the newspaper api
        no_title['title'] = article_title

        #match it by index and replace
        for x in no_title_index:
            df2['title'][x] = no_title['title'][x]
        ```

        The next step was to translate the non-engish text to English using the google api.

        ```python
        #use googletrans api to translate non english titles
        from googletrans import Translator
        import time
        translator = Translator()

        #translate non english ones
        idx_trans = [76, 7317, 7322, 11315, 12054, 13235, 13262]
        list_t = []
        for idx in idx_trans:
            translation = translator.translate(df2['title'][idx], dest='en').text
            list_t.append(translation)
            #google sometimes freezes your api from overuse so use timer
            time.sleep(1)


        #selext rows not english
        trans_df = df2.iloc[df2.index[idx_trans],:]

        #repalce with english Translation
        trans_df['title'] = list_t

        #match it by index and replace
        for x in idx_trans:
            df2['title'][x] = trans_df['title'][x]


        ```

        """
        st.markdown(write_up2, unsafe_allow_html = True)


        image = Image.open('image/foreign.png')
        st.image(image, width = 800)

        write_up3 = """
        The last step is to visualize the data just to make sure it was cleaned right.

        Visualization

            Count
            container_title, publisher, authors, language, subjects
            Numerical Analysis
            title len with score to detect outliers
            publisher with score
            WordCloud
            analysis of title word frequency
        """
        st.markdown(write_up3, unsafe_allow_html = True)

        #Data
        pub_count = pd.read_json('data/pub_count.json')
        container_count = pd.read_json('data/container_count.json')
        lang_count = pd.read_json('data/lang_count.json')
        author_count = pd.read_json('data/author_count.json')
        subj_count = pd.read_json('data/subj_count.json')
        word_count = pd.read_json('data/wordcount.json')
        pub_score = pd.read_json('data/pub_score.json')
        df3 = pd.read_json('data/final.json')

        st.write('First step is to do the Count of some of the columns click the checkbox to see each of the counts.')

        if st.checkbox('Publisher Count '):
            st.write('Publisher Count', pub_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
            pub_count1 = pub_count.head(30)

            fig = go.Figure([go.Bar(x = pub_count1['Publisher Name'], y = pub_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(158,202, 225)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Publisher Name Count', width = 700, height = 1300)
            st.plotly_chart(fig, use_container_width=True)



        if st.checkbox('Container Title Count'):
            st.write('Container Title Count', container_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
            #Plotly barplot of container Count
            container_count1 = container_count.head(30)

            fig = go.Figure([go.Bar(x = container_count1['Container Title'], y = container_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(100,180, 225)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Container Title Count', width = 700, height = 800)

        if st.checkbox('Language Count'):
            st.write('Language Count', lang_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
            #Plotly Language Count
            lang_count1 = lang_count.head(30)

            fig = go.Figure([go.Bar(x = lang_count1['Language'], y = lang_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(302,202, 500)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Language Count', width = 700, height = 800)
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox('Author Count'):
            st.write('Author Count', author_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
                #Plotly Author Viz
            author_count1 = author_count.head(30)

            fig = go.Figure([go.Bar(x = author_count1['Author'], y = author_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(888,202, 88)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Author Count', width = 700, height = 800)
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox('Subject Count'):
            st.write('Subject Count', subj_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
            subj_count1 = subj_count.head(30)

            fig = go.Figure([go.Bar(x = subj_count1['Subject'], y = subj_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(158,10, 225)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Subject Count', width = 700, height = 800)
            st.plotly_chart(fig, use_container_width=True)

        st.write('I then did some analysis on some of the numerical columns. \
        \n \n Which led me to find an outlier with a title that had 43k words.')

        if st.checkbox('Outlier Discovery'):

            image = Image.open('image/outlier.png')
            st.image(image, width = 800)

            st.write('To fix this I used the newspaper3k API again to find the original title and set a limit for 280 words')

            #Now visualize text_len with score and their Publisher
            fig = go.Figure(data=go.Scatter(x=df3['score'],
                                    y=df3['title_len'],
                                    mode='markers',
                                    text=df3['publisher'])) # hover text goes here
            fig.update_layout(title='Publisher With Text length score ')
            st.plotly_chart(fig, use_container_width = True)

            st.write('I also looked to see which publisher had the best score.')
            st.write(pub_score.head(20))
            pub_score1 = pub_score.head(9)
            #Publisher with Score viz

            fig = go.Figure([go.Bar(x = pub_score1['score'], y = pub_score1['publisher'])])
            fig.update_traces(marker_color = 'rgb(158,10, 3)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6, orientation='h')
            fig.update_layout(title_text = 'Publisher Score')
            st.plotly_chart(fig, use_container_width = True)




        nlp = """
        I did some text cleaning with regex as well as preprocessing for the word cloud and the word count.

        ```python
        from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
        from PIL import Image
        from bs4 import BeautifulSoup
        import nltk
        from nltk.corpus import stopwords
        stopwords = set(stopwords.words('english'))
        import matplotlib.pyplot as plt


        #Helper function to clean

        #custom puncutation filter
        punc = lambda x: re.sub("!|,|\?|\'|-|\"|&|。|\)|\(|！|，|\.*|/|\[|\]|\u2026|\d|:|~|、|？|☆|’|– |【|】|「|」|《|》|※| “|”|＊|→||[\b\.\b]{3}||@||@ |#|# |", '',x)

        def clean_text(text):
            #use beautiful soup to remove the &/amps etc in tweets as well as website links
            soup_ = BeautifulSoup(text, 'lxml')
            soup_ = soup_.get_text()
            soup_ = re.sub(r'https?://[A-Za-z0-9./]+', '', soup_)

            #lowercase the words and remove punctuation
            lower_ = ''.join([word.lower() for word in soup_])

            #remove puncutations using a custom list
            punc_ = ''.join([punc(word) for word in lower_])
            #tokenize
            token_ = re.split('\W+',punc_)
            #remove stopwords
            stop_ = [word for word in token_ if word not in stopwords]
            text = ' '.join(word for word in stop_)
            #remove short words for english and translated portions
            text = ' '.join([w for w in text.split() if len(w)>3])

            return text

        #Creating column clean text
        df3['title_clean'] = df3['title'].apply(clean_text)


        stop_words = ['animal', ' animal', 'animal ', 'anima l', '  animal  ']
        #creating word visualization from title
        ele = np.array(Image.open('elephant.jpg'))

        # Read the whole text.
        text_title = ''.join(text for text in df3['title_clean'])
        wc = WordCloud(background_color = 'white', max_words = 1000, width=1600, height=800,\
                              contour_width = 1,contour_color ='black', mask = ele, stopwords=stop_words)
        wc.generate(text_title)

        plt.figure(figsize=(15,8),  facecolor='k')
        plt.imshow(wc,  interpolation ='bilinear')
        plt.axis("off")

        plt.show()
        ```
        """
        st.markdown(nlp, unsafe_allow_html = True)


        if st.checkbox('Word Cloud'):
            image = Image.open('image/word_cloud.png')
            st.image(image, width = 800)

        if st.checkbox('Word Count'):
            st.write('Word Count', word_count.head(30))

        #Plotly Language Count
            #Plotly Publisher Count viz
            word_count1 = word_count[1:].head(30)


            fig = go.Figure([go.Bar(x = word_count1['Word'], y = word_count1['Count'])])
            fig.update_traces(marker_color = 'rgb(158,777, 225)', marker_line_color = 'rgb(8,48,107)',
                             marker_line_width = 1.5, opacity = 0.6)
            fig.update_layout(title_text = 'Word Count', width = 800, height = 800)
            st.plotly_chart(fig, use_container_width=True)

        st.write('The final cleaned dataframe is here. Which you can also download!', df3.head(50))

        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            return f'<a href="data:file/csv;base64,{b64}" download="animalquery.csv">Download csv file</a>'

        st.markdown(get_table_download_link(df3), unsafe_allow_html=True)

#Your own Query
    elif option == 'Do Your Own Simple Query':
        st.title("Let us do our own simple query and get a csv file!")
        html_temp = """
        <div style="background-color:#E31C3C;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">1000 limit Simple Query</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        st.write('coming soon')




if __name__ == '__main__':
    main()
