"""
Author: DHSong
Last Modified: 2020.07.03
Description: Naive Bayes Classification.
"""

import csv
from konlpy.tag import Komoran
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import jpype

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
MAXIMUM_CATEGORY = 500
STOP_WORD = ['!!', '가나', '가부', '경우', '관련', '국민', '기간', '니다', '당장', '등등', '때문', '라고', '만원',
             '배로', '보이', '부여', '부탁', '사람', '사실', '생각', '실상', '아냐', '요새', '요즘', '우리들의',
             '이번', '인물', '입장', '정도', '자신', '저도', '절대', '제가', '조가', '지금', '천원', '포스', '한곳',
             '해먹', '해주시']
DICTIONARY = './data/user_dic.txt'

# Include all the Categories
'''
category_index_dict = {}
category_freq_dict = {}
idx = 0
for category in petition_category:
    if category not in category_index_dict.keys():
        category_index_dict[category] = idx
        idx += 1
        category_freq_dict[category] = 0
'''

# Include limited Categories (top 3 frequency)
category_index_dict = {'정치개혁': 0, '인권/성평등': 1, '안전/환경': 2}
category_freq_dict = {'정치개혁': 0, '인권/성평등': 0, '안전/환경': 0}


komoran = Komoran(userdic=DICTIONARY)
petition_no_used = []
petition_category_used = []
petition_content_used = []

data = []
labels = []

petition_error = 0

print('========\t[ ] Natural Language Processing\t========')
for idx in range(len(petition_no)):
    no = petition_no[idx]
    category = petition_category[idx]
    content = petition_content[idx]

    # Skip Unwanted Category
    if (category not in category_index_dict.keys()) or (category_freq_dict[category] >= MAXIMUM_CATEGORY):
        continue

    # Try - Except for Preparing Unwanted Exception by Komoran
    try:
        words = komoran.nouns(content)
    except:
        petition_error += 1
        continue

    # Make Token and Pre-processing
    tokens = []
    for word in words:
        if word not in STOP_WORD and len(word) >= 2:
            tokens.append(word)

    # Update docs
    data.append(' '.join(tokens))
    labels.append(category_index_dict[category])

    # Update category_freq_dict
    category_freq_dict[category] += 1

    # Update petition_*_used
    petition_no_used.append(no)
    petition_category_used.append(category)
    petition_content_used.append(content)

    # Verbose
    if len(data) % 100 == 0:
        print('Processing %08d Petition Content (%.2f%%) => Valid(%d) : Invalid(%d)' %
              ((idx + 1), ((idx + 1) / len(petition_no)) * 100, len(data), petition_error))

print('========\t[*] Natural Language Processing\t========')


print('\n========\tSummary\t========')
print('> Exception occurs %d times while handling Komoran Noun Extraction.' % petition_error)
print('> The Number of Valid Text is %d' % len(petition_no_used))
print('> Category Information')
for k, v in category_freq_dict.items():
    print('Category: %s\t%s' % (k, str(v)))


print('\n========\t[ ] Save Used Data\t========')
petition_filename_without_error = './data/petition_data_classification_sample.csv'
with open(petition_filename_without_error, encoding='utf', mode='w') as f:
    cout = csv.writer(f)
    cout.writerow(['NO', 'CATEGORY', 'CONTENT'])
    for idx in range(len(petition_no_used)):
        cout.writerow([petition_no_used[idx], petition_category_used[idx], petition_content_used[idx]])
print('========\t[*] Save Used Data\t========')


# tf-idf vectorize and find Main Keyword for each Category
print('\n========\t[ ] Feature Selection Verification\t========')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
data_features = vectorizer.fit_transform(data).toarray()

print('Shape of tf-idf Vectorized Result : [%d x %d]' % (data_features.shape[0], data_features.shape[1]))
print('\n>Most Correlated Keyword by chi-square')
N = 5
for category, category_idx in sorted(category_index_dict.items()):
    labels_binary = []
    for label in labels:
        if label == category_idx:
            labels_binary.append(1)
        else:
            labels_binary.append(0)

    data_features_chi2 = chi2(data_features, labels_binary)
    indices = np.argsort(data_features_chi2[0])
    data_feature_names = np.array(vectorizer.get_feature_names())[indices]
    unigrams = [v for v in data_feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in data_feature_names if len(v.split(' ')) == 2]
    print('Category: %s' % category)
    print('Most correlated unigrams:\t%s' % ' | '.join(unigrams[-N:]))
    print('Most correlated bigrams:\t%s\n' % ' | '.join(bigrams[-N:]))
print('========\t[*] Feature Selection Verification\t========')

print('========\t[ ] Classification using tf-idf\t========')

k = 10
k_fold = StratifiedKFold(n_splits=k)

data_classification = np.array(data)
labels_classification = np.array(labels)

smoothing_factor_option = [1.0, 2.0, 3.0, 4.0, 5.0]

k = 1
for train_indices, test_indices in k_fold.split(data_classification, labels_classification):
    print('%dth validation' % k)
    X_train, X_test = data_classification[train_indices], data_classification[test_indices]
    Y_train, Y_test = labels_classification[train_indices], labels_classification[test_indices]
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
    term_docs_train = vectorizer.fit_transform(X_train)
    term_docs_test = vectorizer.transform(X_test)
    for smoothing_factor in smoothing_factor_option:
        clf = MultinomialNB(alpha=smoothing_factor, fit_prior=True)
        clf.fit(term_docs_train, Y_train)
        prediction = clf.predict(term_docs_test)

        n_correct_dict = {}
        n_error_dict = {}
        for idx in range(len(prediction)):
            if Y_test[idx] == prediction[idx]:
                if Y_test[idx] not in n_correct_dict.keys():
                    n_correct_dict[Y_test[idx]] = 1
                else:
                    n_correct_dict[Y_test[idx]] += 1
            else:
                if Y_test[idx] not in n_error_dict.keys():
                    n_error_dict[Y_test[idx]] = 1
                else:
                    n_error_dict[Y_test[idx]] += 1

        print(category_index_dict)
        for key in sorted(n_correct_dict.keys()):
            print('Accuracy for %d: %.4lf' %
                  (key, float(n_correct_dict[key]) / (n_error_dict[key] + n_correct_dict[key])))
    k += 1

    print('Processing NB')
print('========\t[*] Classification using tf-idf\t========')