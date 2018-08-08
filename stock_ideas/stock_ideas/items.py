# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class StockIdeasItem(scrapy.Item):
    author = scrapy.Field()
    publish_time = scrapy.Field()
    about = scrapy.Field()
    includes = scrapy.Field()
    title = scrapy.Field()
    summary = scrapy.Field()
