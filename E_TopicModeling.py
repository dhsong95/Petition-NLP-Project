from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from konlpy.tag import Komoran
import csv

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
STOP_WORD = ['!!', '가나', '가부', '경우', '관련', '국민', '기간', '니다', '당장', '등등', '때문', '라고', '만원',
             '배로', '보이', '부여', '부탁', '사람', '사실', '생각', '실상', '아냐', '요새', '요즘', '우리들의',
             '이번', '인물', '입장', '정도', '자신', '저도', '절대', '제가', '조가', '지금', '천원', '포스', '한곳',
             '해먹', '해주시']
DICTIONARY = './data/user_dic.txt'

komoran = Komoran(userdic=DICTIONARY)
petition_no_used = []
petition_category_used = []
petition_content_used = []

data = []

petition_error = 0

CATEGORY = '기타'

print('========\t[ ] Natural Language Processing\t========')
for idx in range(len(petition_no)):
    no = petition_no[idx]
    category = petition_category[idx]
    content = petition_content[idx]

    # Skip Unwanted Category
    if category != CATEGORY:
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
print('> Category Information[%s] : %d' % (CATEGORY, len(data)))


print('\n========\t[ ] Save Used Data\t========')
petition_filename_without_error = './data/petition_data_lda_sample.csv'
with open(petition_filename_without_error, encoding='utf', mode='w') as f:
    cout = csv.writer(f)
    cout.writerow(['NO', 'CATEGORY', 'CONTENT'])
    for idx in range(len(petition_no_used)):
        cout.writerow([petition_no_used[idx], petition_category_used[idx], petition_content_used[idx]])
print('========\t[*] Save Used Data\t========')


cv = CountVectorizer(stop_words="english", max_features=1000)
transformed = cv.fit_transform(data)
lda = LatentDirichletAllocation(n_components=3, random_state=43).fit(transformed)

for topic_idx, topic in enumerate(lda.components_):
    label = '{}: '.format(topic_idx)
    print(label, " ".join([cv.get_feature_names()[i]
                           for i in topic.argsort()[:-9:-1]]))


doc_topic = lda.transform(transformed)
for n in range(doc_topic.shape[0]):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}".format(n,topic_most_pr))
