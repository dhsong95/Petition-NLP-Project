"""
Author: DHSong
Last Modified: 2020.07.03
Description: Crawling Data in https://www1.president.go.kr/petitions
"""

import os.path
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

filename_petitions = './data/petitions.csv'
START_NO = 579799           
MAXIMUM_PETITION = 20000
error = 0

# Check if there is crawled data.
is_crawled = False
crawled = []
if os.path.isfile(filename_petitions):
    is_crawled = True
    with open(file=filename_petitions, encoding='utf-8', mode='r') as f:
        cin = csv.reader(f)
        is_first = True
        for row in cin:
            if is_first:
                is_first = False
                continue
            crawled.append(row[0])


# Crawling data.
with open(file=filename_petitions, encoding='utf-8', mode='a') as f:
    cout = csv.writer(f)
    if not is_crawled:
        cout.writerow(['NO', 'TITLE', 'COUNT', 'STATE', 'CATEGORY', 'DATE_START', 'DATE_END', 'CONTENT'])

    for idx in range(MAXIMUM_PETITION):
        url = 'https://www1.president.go.kr/petitions/' + str(START_NO - idx)

        if str(START_NO - idx) in crawled:
            idx += 1
            if idx % 10 == 0:
                print('Number of Page Crawling %06d/%06d (%.2f%%)'
                      % (idx, MAXIMUM_PETITION, (float(idx) / MAXIMUM_PETITION) * 100))
            continue

        page = urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')

        selector_title = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                         'div > h3'
        selector_count = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                         'div > h2 > span'
        selector_state = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                         'div > div.petitionsView_progress > h4'
        selector_category = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                            'div > div.petitionsView_info > ul > li:nth-child(1)'
        selector_date_start = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                              'div > div.petitionsView_info > ul > li:nth-child(2)'
        selector_date_end = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                            'div > div.petitionsView_info > ul > li:nth-child(3)'
        selector_content = '#cont_view > div.cs_area > div.new_contents > div > div.petitionsView_left > ' \
                           'div > div.petitionsView_write > div.View_write'

        try:
            title = soup.select_one(selector_title).text.strip()
            count = soup.select_one(selector_count).text.strip()
            count = count.replace(',', '')
            state = soup.select_one(selector_state).text.strip()
            category = soup.select_one(selector_category).text.strip()
            category = category.replace('카테고리', '')
            date_start = soup.select_one(selector_date_start).text.strip()
            date_start = date_start.replace('청원시작', '')
            date_end = soup.select_one(selector_date_end).text.strip()
            date_end = date_end.replace('청원마감', '')
            content = soup.select_one(selector_content).text.strip()

            # print(title + '\t' + count + '\t' + state + '\t' + category + '\t' + date_start + '\t' + date_end)
            cout.writerow([str(START_NO - idx), title, count, state, category, date_start, date_end, content])

            idx += 1
            if idx % 10 == 0:
                print('Number of Page Crawling %06d/%06d (%.2f%%)'
                      % (idx, MAXIMUM_PETITION, (float(idx) / MAXIMUM_PETITION) * 100))
        except AttributeError:
            error += 1
            continue

    print('\nSuccessful: %06d\tFail:%06d' % (MAXIMUM_PETITION - error, error))
    # Successful: 000130      Fail:019870 (2020 07 04)

