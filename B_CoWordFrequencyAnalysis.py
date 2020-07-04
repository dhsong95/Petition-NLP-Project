"""
Author: DHSong
Last Modified: 2020.07.03
Description: Coword Analysis on Petition about Politics
"""

import csv
import collections
from konlpy.tag import Komoran


# Consider all the possible pair
def pairwise(iterable):
    pair = []
    for i in range(len(iterable)):
        for j in range(i + 1, len(iterable)):
            a = iterable[i]
            b = iterable[j]
            if a != b:
                pair.append((a, b) if a < b else (b, a))

    return pair


# Read File
csv.field_size_limit(100000000)
petition_no = []
petition_category = []
petition_content = []
petition_filename = './data/petitions.csv'
with open(file=petition_filename, encoding='utf=8', mode='r') as f:
    cin = csv.reader(f)
    is_first = True
    for row in cin:
        if is_first:
            is_first = False
            continue
        petition_no.append(row[0])
        petition_category.append(row[4])
        petition_content.append(row[-1])


# Parameter to be used
MAXIMUM_PETITION = 1500
CATEGORY = ['정치개혁']
STOP_WORD = ['!!', '가나', '가부', '경우', '관련', '국민', '기간', '니다', '당장', '등등', '때문', '라고', '만원',
             '배로', '보이', '부여', '부탁', '사람', '사실', '생각', '실상', '아냐', '요새', '요즘', '우리들의',
             '이번', '인물', '입장', '정도', '자신', '저도', '절대', '제가', '조가', '지금', '천원', '포스', '한곳',
             '해먹', '해주시']
DICTIONARY = './data/user_dic.txt'

komoran = Komoran(userdic=DICTIONARY)
petition_no_used = []
petition_category_used = []
petition_content_used = []
docs = []
petition_error = 0
category_freq_dict = {}

print('========\t[ ] Natural Language Processing\t========')
for idx in range(len(petition_no)):
    no = petition_no[idx]
    category = petition_category[idx]
    content = petition_content[idx]

    # Skip Unwanted Category
    if len(CATEGORY) != 0 and category not in CATEGORY:
        continue

    # Try - Except for Preparing Unwanted Exception by Komoran
    try:
        words = komoran.nouns(content)
    except:
        petition_error += 1
        continue

    # Tokenization and Pre-processing
    tokens = []
    for word in words:
        if word not in STOP_WORD and len(word) >= 2:
            tokens.append(word)

    # Update docs
    docs.append(tokens)

    # Update category_freq_dict
    if category not in category_freq_dict.keys():
        category_freq_dict[category] = 1
    else:
        category_freq_dict[category] += 1

    # Update petition_*_used
    petition_no_used.append(no)
    petition_category_used.append(category)
    petition_content_used.append(content)

    # Verbose
    if len(docs) % 100 == 0:
        print('Processing %08d Petition Content (%.2f%%)' % (len(docs), (len(docs) / MAXIMUM_PETITION) * 100))

    if len(docs) >= MAXIMUM_PETITION:
        break
print('========\t[*] Natural Language Processing\t========')

print('\n========\tSummary\t========')
print('> Exception occurs %d times while handling Komoran Noun Extraction.' % petition_error)
print('> The Number of Valid Text is %d' % len(petition_no_used))
print('> Category Information')
for k, v in category_freq_dict.items():
    print('Category: %s\t%s' % (k, str(v)))


print('\n========\t[ ] Save Used Data\t========')
petition_filename_without_error = './data/petition_data_coword_sample.csv'
with open(petition_filename_without_error, encoding='utf', mode='w') as f:
    cout = csv.writer(f)
    cout.writerow(['NO', 'CATEGORY', 'CONTENT'])
    for idx in range(len(petition_no_used)):
        cout.writerow([petition_no_used[idx], petition_category_used[idx], petition_content_used[idx]])
print('========\t[*] Save Used Data\t========')


print('\n========\t[ ] Word-Frequency Data\t========')
docs_flat = []
for doc in docs:
    docs_flat += doc
word_freq = collections.Counter(docs_flat)
word_freq_common = word_freq.most_common()

petition_filename_dictionary = './data/petition_frequency_sample.csv'
with open(file=petition_filename_dictionary, mode='w', encoding='utf-8') as f:
    cout = csv.writer(f)
    cout.writerow(['WORD', 'FREQUENCY'])
    for item in word_freq_common:
        cout.writerow([item[0], item[1]])

TOP = 100
word_freq_common = word_freq.most_common(TOP)
print('> Top %d Word Frequency Pair' % TOP)
for item in word_freq_common:
    print('%s\t:%s' % (item[0], item[1]))
print('========\t[*] Word-Frequency Data\t========')

print('\n========\t[ ] Co-World Analysis\t========')
pairs = []
idx = 0
for doc in docs:
    pair = pairwise(doc)
    pairs += pair

    # Verbose
    idx += 1
    if idx % 100 == 0:
        print('Processing %08d Document (%.2f%%)' % (idx, (idx / len(docs)) * 100))

pair_freq = collections.Counter(list(pairs))

TOP = 100
pair_freq_common = pair_freq.most_common(TOP)
print('\n> Top %d Pair Frequency Pair' % TOP)
for item in pair_freq_common:
    print('%s\t:%s' % (item[0], item[1]))

print('========\t[*] Co-World Analysis\t========')


print('\n========\t[ ] Save Graphml by Networkx\t========')

import networkx as nx

graph = nx.Graph()
idx = 0
for key in pair_freq.keys() :
    graph.add_edge(key[0], key[1], weight=pair_freq.get(key))
    idx += 1
    if idx % 100000 == 0:
        print('Processing %08d key (%.2f%%)' % (idx, (idx / len(pair_freq.keys())) * 100))

petition_filename_coword = './data/petition_coward_sample.graphml'
nx.write_graphml(graph, petition_filename_coword)
print('\n========\t[*] Save Graphml by Networkx\t========')