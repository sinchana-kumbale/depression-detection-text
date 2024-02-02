from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
import requests
import re
import os

URL = "" #Add URL to required webpage with downloads


def get_soup(URL):
	return bs(requests.get(URL).text, 'html.parser')

for link in get_soup(URL).findAll("a", attrs={'href': re.compile(".zip")}):
	if isinstance(link, NavigableString):
		continue
	file_link = link.get('href')
	
	if os.path.isfile(link.text):
		continue
	print(file_link)
	with open(link.text, 'wb') as file:
		response = requests.get(URL + file_link)
		file.write(response.content)
