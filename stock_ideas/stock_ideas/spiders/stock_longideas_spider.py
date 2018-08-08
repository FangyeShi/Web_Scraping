from scrapy import Spider,Request
from stock_ideas.items import StockIdeasItem

class StockShortIdeasSpider(Spider):
    name = 'stock_long_ideas_spider'

    # Anonymous name to prevent scraping by the website. Should be easy to guess by human being.
    # Hint: Website famous for community driven insights on every topic of interest to investors.
    allowed_urls = ['https://sxxxxxxaxxxx.com']
    start_urls = ['https://sxxxxxxaxxxx.com/stock-ideas/long-ideas?page=1']



    def parse(self, response):

        urls = response.xpath('//a[@class="a-title"]/@href').extract()
        root_url = 'https://sxxxxxxaxxxx.com'

        # Note: each URL found above is a relative URL. So need to join it with root_url.
        for url in urls:
            # Yield articles to be parsed.
            yield Request(url=root_url + url,callback=self.parse_article_page)

        next_page = response.xpath('//li[@class="next"]/a/@href').extract_first()
        if next_page is not None:
            # If there is a next_page, go there and repeat above steps using recursive call.
            yield Request(next_page, callback=self.parse)

    def parse_article_page(self,response):

        # Get all the entries of interest to us.
        title = response.xpath('//h1[@itemprop="headline"]/text()').extract_first()
        publish_time = response.xpath('//time[@itemprop="datePublished"]/@content').extract_first()
        about = response.xpath('//span[@id="about_primary_stocks"]/a/@href').extract()
        # about should be a stock ticker or several stock tickers
        # If about is nonempty, we join each stock tickers by '||' else about is just empty string.
        if about:
            about = [symbol[8:] for symbol in about]
            about = '||'.join(about)
        else:
            about = ''
        # Same methodology as 'about' entry.
        includes = response.xpath('//span[@id="about_stocks"]/a/@href').extract()
        if includes:
            includes = [symbol[8:] for symbol in includes]
            includes = '||'.join(includes)
        else:
            includes = ''
        author = response.xpath('//span[@itemprop="name"]/text()').extract_first()

        # Join paragraphs of summary by '\n'
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
