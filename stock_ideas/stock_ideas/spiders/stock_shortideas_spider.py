from scrapy import Spider,Request
from stock_ideas.items import StockIdeasItem

class StockShortIdeasSpider(Spider):
    name = 'stock_short_ideas_spider'

    # Anonymous name to prevent scraping by the website. Should be easy to guess by human being.
    # Hint: Website famous for community driven insights on every topic of interest to investors.
    allowed_urls = ['https:///sxxxxxxaxxxx.com']
    start_urls = ['https:///sxxxxxxaxxxx.com/stock-ideas/short-ideas?page=1']



    def parse(self, response):

        urls = response.xpath('//a[@class="a-title"]/@href').extract()
        root_url = 'https://sxxxxxxaxxxx.com'

        for url in urls:
            yield Request(url=root_url + url,callback=self.parse_article_page)

        next_page = response.xpath('//li[@class="next"]/a/@href').extract_first()
        if next_page is not None:
            yield Request(next_page, callback=self.parse)

    def parse_article_page(self,response):

        title = response.xpath('//h1[@itemprop="headline"]/text()').extract_first()
        publish_time = response.xpath('//time[@itemprop="datePublished"]/@content').extract_first()

        about = response.xpath('//span[@id="about_primary_stocks"]/a/@href').extract()
        if about:
            about = [symbol[8:] for symbol in about]
            about = '||'.join(about)
        else:
            about = ''

        includes = response.xpath('//span[@id="about_stocks"]/a/@href').extract()
        if includes:
            includes = [symbol[8:] for symbol in includes]
            includes = '||'.join(includes)
        else:
            includes = ''

        author = response.xpath('//span[@itemprop="name"]/text()').extract_first()

        summary = response.xpath('//div[@class="article-summary article-width"]/div[1]/p/text()').extract()
        summary = '\n'.join(summary)


        item = StockIdeasItem()
        item['author'] = author
        item['publish_time'] = publish_time
        item['about'] = about
        item['includes'] = includes
        item['title'] = title
        item['summary'] = summary

        yield item
