### This approach doesn't work because of the anti-scraping mechanism
### deployed by the website : It will redirect you to a captcha page
### very soon. However, the skeleton of the code is included anyway.
### I think if there is a way to add random user agents and random proxies,
### this could work.

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time
import csv
import re

# Windows users need to specify the path to chrome driver.
driver = webdriver.Chrome(r'path to chromedriver.exe')

#Mac users just call the following:
#driver = webdriver.Chrome()

driver.get("https://sxxxxxxaxxxx.com/stock-ideas/long-ideas")
# Or: driver.get("https://sxxxxxxaxxxx.com/stock-ideas/short-ideas")

# For Windows OS, adding newline='' will prevent empty rows from being added to the csv file.
csv_file = open('long.csv', 'w',newline='')

# For Mac:
#csv_file = open('long.csv', 'w')

writer = csv.writer(csv_file)

index = 1

while True:
	try:
		print("Scraping Page number " + str(index))
		index = index + 1

		# Find all the reviews on the page
		wait_articles = WebDriverWait(driver, 10)
		articles = wait_articles.until(EC.presence_of_all_elements_located((By.XPATH,
									'//a[@class="a-title"]')))

		# time.sleep(2) Adding random sleep here and there is not helpful. Still get banned :(

		# Here, I try to open new article in a new window. It is not necessary but
		# it is nice to know how to do it anyway.
		# In order to do that, I need to save current window as main_window.
		main_window = driver.current_window_handle

		for article in articles:

			absolute_url = article.get_attribute('href')
			# Note: when using selenium, we will get absolute url despite
			# the fact the the href attribute only gives relative url.

			# Here I open a new window.
			driver.execute_script('window.open(arguments[0]);',absolute_url)

			# Locate where the new window is.
			new_window = [window for window in driver.window_handles if window != main_window][0]

			# Switch to new window.
			driver.switch_to.window(new_window)


			article_dict = {}

			title = driver.find_element_by_xpath('//h1[@itemprop="headline"]').text

			publish_time = driver.find_element_by_xpath('//time[@itemprop="datePublished"]').get_attribute('content')

			about = driver.find_elements_by_xpath('//span[@id="about_primary_stocks"]/a')
			about = [ticker.get_attribute('href') for ticker in about]
			if about:
				# Because we are getting absolute url, we need to slice after 32nd element to get the stock ticker.
				about = [symbol[32:] for symbol in about]
				about = '||'.join(about)
			else:
				about = ''

			includes = driver.find_elements_by_xpath('//span[@id="about_stocks"]/a')
			includes = [ticker.get_attribute('href') for ticker in includes]
			if includes:
				includes = [symbol[32:] for symbol in includes]
				includes = '||'.join(includes)
			else:
				includes = ''

			author = driver.find_element_by_xpath('//span[@itemprop="name"]').text

			summary = driver.find_elements_by_xpath('//div[@class="article-summary article-width"]/div[1]/p')
			summary = [paragraph.text for paragraph in summary]
			if summary:
				summary = [s.strip() for s in summary]
				summary = '\n'.join(summary)
			else:
				summary = ''

			article_dict['author'] = author
			article_dict['publish_time'] = publish_time
			article_dict['about'] = about
			article_dict['includes'] = includes
			article_dict['title'] = title
			article_dict['summary'] = summary

			writer.writerow(article_dict.values())

			# time.sleep(2)

			# Close current tab.
			driver.close()

			# time.sleep(2)

			# Going back to main window.
			driver.switch_to.window(main_window)

			# time.sleep(2)


		# Locate the next button on the page.
		wait_button = WebDriverWait(driver, 10)
		next_button = wait_button.until(EC.element_to_be_clickable((By.XPATH,
									'//li[@class="next"]/a')))
		# Go to next page:
		next_button.click()
	except Exception as e:
		print(e)
		csv_file.close()
		driver.close()
		break
