# -*- coding: utf-8 -*-

# Scrapy settings for stock_ideas project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'stock_ideas'

SPIDER_MODULES = ['stock_ideas.spiders']
NEWSPIDER_MODULE = 'stock_ideas.spiders'


USER_AGENT_LIST = [ a list of most commonly used user agents ]
# Can get a list from here: https://techblog.willshouse.com/2012/01/03/most-common-user-agents/

HTTP_PROXY_LIST = [ a list of most recent free proxies that support HTTPS ]
# Each time you crawl , you should get a list of most recent/verified free proxies to
# ensure maximal probability of success.
# Can get a list from here: https://free-proxy-list.net/ .

DOWNLOADER_MIDDLEWARES = {
     'stock_ideas.middlewares.RandomUserAgentMiddleware': 400,
     'stock_ideas.middlewares.ProxyMiddleware': 410,
     'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None
     # Disable default UserAgentMiddleware so that our list of user agents will be used.
}

RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 403]
RETRY_TIMES = 3 # This number could be any number you want. The larger, the longer
                # the spiders will crawl without getting code 403 and shut down.

# Unfortunately, cannot obey robots.txt rules unless we don't want to scrape
# that website :(
ROBOTSTXT_OBEY = False


#DOWNLOAD_DELAY = 3
#Disable cookies (enabled by default):
#COOKIES_ENABLED = False


# Enable autothrottle will slow down the spiders a bit but increase stability a lot.
AUTOTHROTTLE_ENABLED = True

ITEM_PIPELINES = {'stock_ideas.pipelines.StockIdeasPipeline': 300}
