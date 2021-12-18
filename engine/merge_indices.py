# merge_indices.py

import os
import math
import json
import heapq

from create_indices import listdir_nohidden, save_index


def merge_postings(first: list, second: list):
    p1 = 0
    p2 = 0

    new_posting = []
    while p1 < len(first) and p2 < len(second):
        if first[p1]['id'] < second[p2]['id']:
            new_posting.append(first[p1])
            p1 += 1
        else:
            new_posting.append(second[p2])
            p2 += 1
    
    if p1 < len(first):
        new_posting.extend(first[p1: ])
    if p2 < len(second):
        new_posting.extend(second[p2: ])

    return new_posting


def merge_indices():
    # extract indices
    files = listdir_nohidden(PARTIAL_INDICES)
    for index, file in enumerate(files):
        files[index] = os.path.join(PARTIAL_INDICES, file)

    # open pointers to all files
    file_pointers = []
    for file in files:
        file_pointer = open(file, 'r')
        file_pointers.append(file_pointer)
    
    # perform n-way merge, output into a single file
    with open(T_INDEX, 'w') as ti:
        ti.writelines(heapq.merge(*file_pointers))

    # close pointers to all files
    for file_pointer in file_pointers:
        file_pointer.close()

    # delete all partial indices
    for file in files:
        os.remove(file)
    
    os.rmdir(PARTIAL_INDICES)

    # merge lines in T_INDEX
    with open(T_INDEX, 'r') as ti, open(TT_INDEX, 'w') as tti:
        line = ti.readline()
        prev_token = line.split()[0]
        prev_postings = json.loads(line[len(prev_token) + 1: ])

        line = ti.readline()
        while line:
            cur_token = line.split()[0]
            cur_postings = json.loads(line[len(cur_token) + 1: ])

            if cur_token == prev_token:
                prev_postings = merge_postings(cur_postings, prev_postings)
            else:
                tti.write(f'{prev_token} {json.dumps(prev_postings)}\n')
                prev_token = cur_token
                prev_postings = cur_postings

            line = ti.readline()
        
        tti.write(f'{prev_token} {json.dumps(prev_postings)}\n')
        os.remove(T_INDEX)


def add_scores():
    with open(TT_INDEX, 'r') as tti, open(INDEX, 'w') as i, open(PAGERANK, 'r') as pr:
        pagerank = json.load(pr)

        for line in tti:
            token = line.split()[0]
            postings = json.loads(line[len(token) + 1: ])

            idf = math.log10(N / len(postings))
            for posting in postings:
                # calculating tfidf 
                tfidf = posting['tf'] * idf

                # boosting score if twogram 
                tg = 1
                if '_' in token:
                    tg = 2

                # getting pagerank of document
                pr = pagerank[posting['id']]

                # calculating overall score
                score = tfidf * tg * pr * posting['fi']
                posting['se'] = round(score, 2)

                del posting['tf']
                del posting['fi']

            i.write(f'{token} {json.dumps(postings)}\n')

    os.remove(TT_INDEX)
    os.remove(PAGERANK)


def create_seek():
    with open(INDEX, 'r') as i, open(SEEK, 'w') as s:
        seek = dict()
        position = 0

        line = '?'
        while line:
            token = line.split()[0]
            seek[token] = position

            position = i.tell()
            line = i.readline()
        
        del seek['?']

        json.dump(seek, s)
        

def main():
    # merge_indices()
    # add_scores()
    create_seek()


if __name__ == '__main__':
    PARTIAL_INDICES = 'documents/partial_indices'
    T_INDEX = 'documents/t_index.txt'
    TT_INDEX = 'documents/tt_index.txt'
    INDEX = 'documents/index.txt'

    PAGERANK = 'documents/pagerank.json'
    SEEK = 'documents/seek.json'

    # total number of cached sites
    N = 55393

    main()