# create_indices.py

import re
import os
import math
import json
import nltk

from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer

from bs4 import BeautifulSoup


def listdir_nohidden(path: str):
    return [f for f in os.listdir(path) if not f.startswith('.')]


def save_index(inverted_index: dict, path: str):
    with open(path, 'w', encoding='utf-8') as ii:
        sorted_tokens = sorted(inverted_index.keys())
        for token in sorted_tokens:
            ii.write(f'{token} {json.dumps(inverted_index[token])}\n')


def get_twograms(tokens: list):
    return ['_'.join(twogram) for twogram in ngrams(tokens, 2)]


def process_text(text: str, n: int = 10_000_000):
    if not text:
        return []

    # replacement pairs
    REP = {"'": '', ".": '', ",": ' ', "/": ' '}
    REP = dict((re.escape(k), v) for k, v in REP.items()) 
    PATTERN = re.compile("|".join(REP.keys()))

    # replace text according to specifications in PATTERN
    text = PATTERN.sub(lambda m: REP[re.escape(m.group(0))], text)

    # tokenize text
    tokens = []
    if len(text) > n:
        split_text = []
        for index in range(0, len(text), n):
            split_text.append(text[index: index + n])

        split_tokens = []
        for text in split_text:
            split_tokens.append(nltk.word_tokenize(text))
        
        for li_tokens in split_tokens:
            tokens.extend(li_tokens)
    else:
        tokens = nltk.word_tokenize(text)

    # flagging tokens that are not in english or alphanumeric
    for index, token in enumerate(tokens):
        if not token.isascii() or not token.isalnum():
            tokens[index] = '?'
    
    # stemming tokens
    STEMMER = SnowballStemmer('english')

    stems = []
    for token in tokens:
        if token != '?':
            stems.append(STEMMER.stem(token))
    
    # creating twograms from stems
    twograms = get_twograms(stems)
    stems.extend(twograms)

    return stems


def main():
    # path to sites cached by crawler
    CACHED_SITES = 'cached_sites'

    # path to folder where indices will be saved
    PARTIAL_INDICES = 'documents/partial_indices'

    # associate URLs to IDs
    doc_lookup = dict()

    # info necessary for pagerank 
    outgoinglinks_counter = dict()
    incominglinks_lookup = dict()

    index_counter = 0
    posting_counter = 0
    inverted_index = dict()

    # extracting json files from CACHED_SITES
    folders = listdir_nohidden(CACHED_SITES)

    sites = []
    for folder in folders:
        folder_path = os.path.join(CACHED_SITES, folder)
        files = listdir_nohidden(folder_path)

        for index, file in enumerate(files):
            files[index] = os.path.join(folder_path, file)

        sites.extend(files)
    
    # creating PARTIAL_INDICES directory
    if not os.path.exists(PARTIAL_INDICES):
        os.makedirs(PARTIAL_INDICES)

    # creating lookup for documents, building inverted index
    print('total: ', len(sites))
    for doc_ID, site in enumerate(sites):
        doc_ID = str(doc_ID)
        
        # sanity check
        if (doc_ID % 1_000) == 0:
            print('at: ', doc_ID)

        with open(site, 'r', encoding='utf-8') as s:
            document = json.load(s)
            doc_lookup[doc_ID] = document['url']

            content = document['content']
            soup = BeautifulSoup(content, 'lxml')

            # gathering, processing text
            text = soup.get_text()
            tokens = process_text(text)

            # calculating token frequencies
            tf = dict()
            for token in tokens:
                if token in tf:
                    tf[token] += 1
                else:
                    tf[token] = 1

            # normalizing token frequencies
            for token, frequency in tf.items():
                tf[token] = 1 + math.log10(frequency)

            # searching for 'important' text
            fields = dict()
            for token in tf:
                fields[token] = 0

            tags = ['title', 'strong', 'b', 'h1', 'h2', 'h3']
            for tag in tags:
                elements = soup.find_all(tag)
                for element in elements:
                    text = element.get_text()
                    tokens = process_text(text)

                    for token in tokens:
                        if token in fields:
                            fields[token] += 1

            # normalizing fields
            for token, value in fields.items():
                if value == 0:
                    fields[token] = 1
                elif value == 1:
                    fields[token] = 1.2
                else:
                    fields[token] = 1 + math.log10(value)

            # gathering links for pagerank calculations
            links_found = set()
            elements = soup.find_all('a', href=True)
            for element in elements:
                link = element['href']

                # ensuring site doesn't link to self
                if link != document['url']:
                    # ensuring multiple incoming links are counted as one
                    links_found.add(link)

            # building adjacency list from links
            if len(links_found) > 0:
                outgoinglinks_counter[doc_ID] = len(links_found)

                # creating adjacency list from links
                for link in links_found:
                    if link in incominglinks_lookup:
                        incominglinks_lookup[link].append(doc_ID)
                    else:
                        incominglinks_lookup[link] = [doc_ID]
            
            # inserting into index
            for token, frequency in tf.items():
                posting = {
                    'id': doc_ID,
                    'tf': frequency,
                    'fi': fields[token]
                }

                if token in inverted_index:
                    inverted_index[token].append(posting)
                else:
                    inverted_index[token] = [posting]
                posting_counter += 1

                # number of allowed postings per index before the index is saved
                NUM_POSTINGS = 100_000
                if posting_counter >= NUM_POSTINGS:
                    file_name = os.path.join(PARTIAL_INDICES, f'index_{index_counter}.txt')
                    save_index(inverted_index, file_name)

                    index_counter += 1
                    posting_counter = 0
                    inverted_index = dict()
    
    if inverted_index:
        file_name = os.path.join(PARTIAL_INDICES, f'index_{index_counter}.txt')
        save_index(inverted_index, file_name)
    
    # saving doc_lookup
    with open('documents/doc_lookup.json', 'w') as dl:
        json.dump(doc_lookup, dl)
    
    # ensuring links are within the scope of ics.uci
    inverted_doc_lookup = dict()
    for key, val in doc_lookup.items():
        inverted_doc_lookup[val] = key
    
    rm_keys = []
    add_items = []
    for key, val in incominglinks_lookup.items():
        if key not in inverted_doc_lookup:
            for doc_ID in val:
                outgoinglinks_counter[doc_ID] -= 1
        else:
            doc_ID = inverted_doc_lookup[key]
            add_items.append((doc_ID, val))

        rm_keys.append(key)

    for item in add_items:
        incominglinks_lookup[item[0]] = item[1]

    for key in rm_keys:
        del incominglinks_lookup[key]

    # five iterations of page rank
    NUM_ITERATIONS = 5

    pagerank = dict()
    for doc_ID in doc_lookup.keys():
        pagerank[doc_ID] = 1

    d = 0.85    # damping factor (typically between 0.8 and 0.9)
    for iter in range(NUM_ITERATIONS):
        temp_pagerank = dict()
        for doc_ID in pagerank.keys():
            s = 0   # sum
            if doc_ID in incominglinks_lookup:
                incominglinks = incominglinks_lookup[doc_ID]
                for temp_ID in incominglinks:
                    s += (pagerank[temp_ID] / outgoinglinks_counter[temp_ID])

            temp_pagerank[doc_ID] = 0.15 + (d * s)
        pagerank = dict(temp_pagerank)
    
    # saving pagerank
    with open('documents/pagerank.json', 'w') as pr:
        json.dump(pagerank, pr)


if __name__ == '__main__':
    nltk.download('punkt')
    main()