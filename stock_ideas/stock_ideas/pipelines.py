# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.exporters import CsvItemExporter

class StockIdeasPipeline(object):
    def __init__(self):
        self.filename1 = 'long.csv'
        self.filename2 = 'short.csv'
    def open_spider(self, spider):

        # Must use 'wb' mode in Windows OS!
        # This will create additional empty rows in between each observations.
        # However, when using pandas read_csv function, the empty rows will
        # automatically be removed. So this is not a problem at all.
        if spider.name == 'stock_long_ideas_spider':
            self.csvfile = open(self.filename1, 'wb')
        else:
            self.csvfile = open(self.filename2, 'wb')
        self.exporter = CsvItemExporter(self.csvfile)
        self.exporter.start_exporting()
    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.csvfile.close()
    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item
