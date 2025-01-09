from scrapy.crawler import CrawlerProcess
from woocommerce import API
import argparse
import scrapy

parser = argparse.ArgumentParser()
parser.add_argument('url')
args = parser.parse_args()

wcm = API(    
    url='url',
    consumer_key='ck_',
    consumer_secret='cs_',
    version='wc/v3',
    timeout=600
)


def get_categories(wcm):
    per_page = 100
    page = 1
    categories = []

    while True:
        params = {'per_page': per_page, 'page': page}
        response = wcm.get('products/categories', params=params)
        if not response.ok:
            break
        current_categories = response.json()
        if not current_categories:
            break
        categories.extend(current_categories)
        page += 1

    return categories


class SantechParserSpider(scrapy.Spider):
    name = 'santech_parser'
    allowed_domains = ['domen.ru']
    start_urls = ['domen.ru']
    url = args.url 
    pages_count = 1
    # useragent = UserAgent()
    custom_settings = {
        'FEEDS': { 'products.csv': { 'format': 'csv', 'overwrite': True}},
        'DOWNLOAD_DELAY': 0.4
    }
    attributes_count = 0
    categories = get_categories(wcm)
    products = []


    

    def start_requests(self):
        yield scrapy.Request(self.url, callback=self.parse_pages_count)


    def parse_pages_count(self, response):
        try:
            self.pages_count = int(response.xpath('//div[@class = "lk-notifications__pagination_items"]/a/text()').extract()[-1])
    
        except:
            pass

        for page in range(1, self.pages_count + 1):
            url = f'{self.url}?PAGEN_6={page}'
            self.log(url)
            yield scrapy.Request(url, callback=self.parse_url)


    def parse_url(self, response, **kwargs):
        for href in response.xpath('//a[@class = "header__menu_special-title"]/@href'):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback=self.parse)


    def parse(self, response):
        category_id = None
        category_name = response.xpath('//span[@itemprop = "name"]/text()').extract()[-2]
        parent_category = response.xpath('//span[@itemprop = "name"]/text()').extract()[1]

        for category in self.categories:
            if category['name'] == category_name:
                category_id = category['id']
                break

        try:
            if category_id is None:
                parent_category_id = None
                new_category_data = None

                for category in self.categories:
                    if category['name'] == parent_category:
                        parent_category_id = category['id']
                        break

                if parent_category_id is None:
                    parent_category_data = {
                        'name': parent_category,
                    }
                    
                    parent_category_id = wcm.post('products/categories', parent_category_data).json()['id']

                    new_category_data = {
                        'name': category_name,
                        'parent': parent_category_id

                    }

                else:
                    new_category_data = {
                        'name': category_name,
                        'parent': parent_category_id

                    }

                self.log(wcm.post('products/categories', new_category_data).json())

                self.categories = get_categories(wcm)

                for category in self.categories:
                    if category['name'] == category_name:
                        category_id = category['id']
                        break

        except Exception as error:
            self.log(error)

        try:
            item = {
                'sku': response.xpath('//div[@class = "card-tabs__item_row"]/div[2]/text()').extract()[0],
                'name': response.xpath('//h1[@class = "card-info__title"]/text()').extract_first('').strip(),
                'categories': [{'id': category_id}],
                'images': [{'src': 'https://santeh-kirov.ru' + response.xpath('//a[@class = "card-img"]/@href').extract()[0]}],
                'description': response.xpath('//div[@class = "card-tabs__item_text "]/text()').extract()[0].strip(),
                'catalog_visibility': 'visible',
                'regular_price': response.xpath('//div[@class = "card-info__price_value"]/text()').extract()[0].strip(),
                'attributes': []
            }

            regular_price = item['regular_price']

            number, currency = regular_price.replace(' ', '', 1).split()
            number = float(number.replace(' ', ''))
            number *= 0.4
            number = '{:.2f}'.format(number)
            item['regular_price'] = number + ' ' + currency

            self.log(item['regular_price'])

        # except Exception as error:
        # self.log(error)
        except:
            item = {
                'sku': response.xpath('//div[@class = "card-tabs__item_row"]/div[2]/text()').extract()[0],
                'name': response.xpath('//h1[@class = "card-info__title"]/text()').extract_first('').strip(),
                'categories': [{'id': category_id}],
                'images': [{'src': 'https://domen.ru' + response.xpath('//a[@class = "card-img"]/@href').extract()[0]}],
                'description': response.xpath('//div[@class = "card-tabs__item_text "]/text()').extract()[0].strip(),
                'catalog_visibility': 'hidden',
                'regular_price': 'Под заказ',
                'attributes': []
            }

        if 'нет' in response.xpath('//div[@class = "card-info__table_col"]/text()').extract()[-2].strip():
            item['stock_status'] = 'outofstock'

        else:
            item['stock_status'] = 'instock'
        
        attribute_names = response.xpath('//div[@class = "card-tabs__item_table"]/div/div[1]/text()').extract()

        for attribute_name in attribute_names:
            attribute_num = attribute_names.index(attribute_name)

            item['attributes'].append({'name': attribute_name, 'visible': True}) 

        attribute_values = response.xpath('//div[@class = "card-tabs__item_table"]/div/div[2]/text()').extract()

        for attribute_value in attribute_values:
            attribute_num = attribute_values.index(attribute_value)

            item['attributes'][attribute_num]['options'] = [attribute_value]

        # wcm.post('products', item).json()
        if item['categories'][0]['id'] == category_id:
            self.products.append(item)

        else:
            pass

        yield item


    def closed(self, reason):

        self.log(reason)

        batch_size = 90

        batches = [self.products[i:i+batch_size] for i in range(0, len(self.products), batch_size)]

        for batch in batches:
            self.log(wcm.post('products/batch', data={'create': batch}).json())


 
if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(SantechParserSpider)
    process.start()
    
    print('<p align="center"><font color=green>Parsed!<font></p>')
