# _*_ coding: utf-8 _*_
import threading
from argparse import ArgumentParser
from crawling_base import Crawling
from craw_aws import CrawAWS
from craw_seoul_aqi import CrawSeoulAQI
from craw_weather import CrawlWeather
# single thread that calls normal function https://www.tutorialspoint.com/python/python_multithreading.htm
# thread that calls class's function https://stackoverflow.com/questions/15365406/run-class-methods-in-threads-python
class CrawlingSingle(Crawling):

    def __init__(self):
        self.aws = CrawAWS()
        self.aqi = CrawSeoulAQI()
        self.china_aws = CrawlWeather()
    
    def execute(self, args):
        t1 = threading.Thread(target=self.aws.execute, args=(args,))
        t2 = threading.Thread(target=self.aqi.execute, args=(args,))
        t3 = threading.Thread(target=self.china_aws.get_future, args=(args,))
        try:
            t1.start()
            t2.start()   
            t3.start()
            t1.join()
            t2.join() 
            t3.join()        
            print("Threads started")
        except Exception as e:
            print(e)
    
if __name__ == "__main__":
    cs = CrawlingSingle()
    args = cs.add_argument()
    cs.execute(args)


