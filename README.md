# Web_Scraping

*In this project, I scraped the data from SeekxxxAlpxx using python package scrapy.*

*Specifically, I scraped author, publish date, target stock(s), title and summary of roughly 7k articles on long ideas and 10k articles on short ideas*

Note: the span of publish dates is much wider for short ideas.

In order to circumstance the anti-scraping measures deployed by that website, I tried using TOR. But the attempt failed. I suspect that TOR proxies are well-known and they are blocked by the website. (A human user could solve the captcha to continue using the website but our spiders certainly cannot do that.) 

In the end, I used a pool of user agents and proxies and draw randomly from the pools for each request. And this approach worked.


*The code for scrapy spiders can be found in stock_ideas folder.*

*long.csv and short.csv are the data scraped.*

*daily_adjusted_*.csv are csv files for historical daily adjusted prices obtained from* [here](https://www.alphavantage.co/)

Note: for only 4 stocks, I just used the web based API call.

*using_selenium.py is the code for scraping using selenium.* 

Note: for this website, it will get redirected to captcha page very soon and so this is *not* working.

*The python notebook includes data cleaning and data analysis. See presentation.pdf for more accessible/readable analysis.* 
