# _*_ coding: utf-8 _*_
import subprocess
from lxml import etree
import requests
from argparse import ArgumentParser
import math
import properties as pr

class Crawling(object):
    
    def mine_data(self, html):
        return ""
    
    def craw_data_controller(self):
        return ""

    def craw_data(self):
        return ""
    
    # format hour, day < 10 to 10 format
    def format10(self, no):
        if no < 10:
            return "0" + str(no)
        else:
            return str(no)
    
    # write data crawled to file
    def write_log(self, output):
        if output:
            with open(self.filename, "a") as f:
                f.write(output)

    
    # convert pm10 micro value to aqi value
    def AQIPM10(self, Concentration):
        Conc = float(Concentration)
        c = math.floor(Conc)
        if (c >= 0 and c < 55):
            AQI = self.Linear(50, 0, 54, 0, c)
        elif(c >= 55 and c < 155):
            AQI = self.Linear(100, 51, 154, 55, c)
        elif(c >= 155 and c < 255):
            AQI = self.Linear(150, 101, 254, 155, c)
        elif(c >= 255 and c < 355):
            AQI = self.Linear(200, 151, 354, 255, c)
        elif(c >= 355 and c < 425):
            AQI = self.Linear(300, 201, 424, 355, c)
        elif(c >= 425 and c < 505):
            AQI = self.Linear(400, 301, 504, 425, c)
        elif(c >= 505 and c < 605):
            AQI = self.Linear(500, 401, 604, 505, c)
        else:
            AQI = 0
        return AQI


    # convert pm25 micro value to aqi value
    def AQIPM25(self, Concentration):
        Conc = float(Concentration)
        c = (math.floor(10 * Conc)) / 10;
        if (c >= 0 and c < 12.1):
            AQI = self.Linear(50, 0, 12, 0, c)
        elif (c >= 12.1 and c < 35.5):
            AQI = self.Linear(100, 51, 35.4, 12.1, c)
        elif (c >= 35.5 and c < 55.5):
            AQI = self.Linear(150, 101, 55.4, 35.5, c)
        elif (c >= 55.5 and c < 150.5):
            AQI = self.Linear(200, 151, 150.4, 55.5, c)
        elif (c >= 150.5 and c < 250.5):
            AQI = self.Linear(300, 201, 250.4, 150.5, c)
        elif (c >= 250.5 and c < 350.5):
            AQI = self.Linear(400, 301, 350.4, 250.5, c)
        elif (c >= 350.5 and c < 500.5):
            AQI = self.Linear(500, 401, 500.4, 350.5, c)
        else:
            AQI = 0
        return AQI

    def Linear(self, AQIhigh, AQIlow, Conchigh, Conclow, Concentration):
        Conc = float(Concentration)
        a = ((Conc - Conclow) / (Conchigh - Conclow)) * (AQIhigh - AQIlow) + AQIlow
        linear = round(a)
        return linear

    def init_env(self):
        subprocess.call("source activate tensorflow")
    
    def add_argument(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--forward", default=1, type=int, help="continuously collecting flag")
        parser.add_argument("-i", "--interval", default=1, type=int, help="interval time to activate crawling service")
        parser.add_argument("-si", "--save_interval", default=5, type=int, help="interval time to save data")
        # parser.add_argument("-s", "--start", default="2009-03-01 00:00:00", type=str)
        parser.add_argument("-s", "--start", default="2018-10-15 00:00:00", type=str, help="the start crawling point")
        parser.add_argument("-c", "--city", type=str, default="seoul,beijing,shenyang", help="crawl data of a city")
        parser.add_argument("-e", "--end", type=str, help="the end crawling point")
        args = parser.parse_args()
        return args