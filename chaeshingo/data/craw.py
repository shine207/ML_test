import requests
import re
import wget
import time

num = 1
def spider(max_pages):
    page = 120
    while page <= max_pages:
        url="http://www.baduk.or.kr/info/gibo_list.asp?pageNo="+str(page)
        source_code = requests.get(url)
        plain_text = source_code.text

        pattern = 'gibo_load\(\'[^\']*\''
        r = re.compile(pattern) 
        matches = r.finditer(plain_text)
        for m in matches:
            startIndex = m.start()
            endIndex = m.end()
            try:
                print(plain_text[startIndex+11:endIndex-1].encode('utf-8').decode('utf-8'))
            except:
                print('encode error')
            try:
                wget.download(plain_text[startIndex+11:endIndex-1])
            except:
                print('Error')

            time.sleep(5)
            '''
            gibo = urllib.urlopen(plain_text[startIndex+11:endIndex-1])
            with open(str(num)+".gibo", 'wb') as output:
                output.write(gibo.read())
            '''

        time.sleep(30)
        page += 1


spider(1400)




