"""
craw weather from https://www.worldweatheronline.com/daegu-weather-history/kr.aspx
"""

# _*_ coding: utf-8 _*_
from bs4 import BeautifulSoup as Soup
import requests
from argparse import ArgumentParser
from datetime import datetime, timedelta
import utils
import properties as pr


# clear out html and get data
def mine_data(date, html):
    tables = html.find('div', attrs={"class": "weather_tb tb_years tb_years_8"})
    weathers = tables.find('div', attrs={"class": "tb_row tb_weather"}).find_all("div", attrs={"class": "tb_cont_item"})
    temps = tables.find('div', attrs={"class": "tb_row tb_temp"}).find_all("div", attrs={"class": "tb_cont_item"})
    feels = tables.find('div', attrs={"class": "tb_row tb_feels"}).find_all("div", attrs={"class": "tb_cont_item"})
    winds = tables.find('div', attrs={"class": "tb_row tb_wind"}).find_all("div", attrs={"class": "tb_cont_item"})
    gusts = tables.find_all('div', attrs={"class": "tb_row tb_gust"})
    dirs = None
    if len(gusts) > 1:
        dirs = gusts[0].find_all("div", attrs={"class": "tb_cont_item"})
        gusts = gusts[1].find_all("div", attrs={"class": "tb_cont_item"})
    else:
        gusts = gusts[0].find_all("div", attrs={"class": "tb_cont_item"})
    clouds = tables.find('div', attrs={"class": "tb_row tb_cloud"}).find_all("div", attrs={"class": "tb_cont_item"})
    humids = tables.find('div', attrs={"class": "tb_row tb_humidity"}).find_all("div", attrs={"class": "tb_cont_item"})
    preps = tables.find('div', attrs={"class": "tb_row tb_precip"}).find_all("div", attrs={"class": "tb_cont_item"})
    pressures = tables.find('div', attrs={"class": "tb_row tb_pressure"}).find_all("div", attrs={"class": "tb_cont_item"})
    values = []
    i = 0
    for wt,t,f,w,g,c,h,r,p in zip(weathers,temps,feels,winds,gusts,clouds,humids,preps,pressures):
        if i:
            w_ = wt.find("img")["alt"]
            t_ = t.get_text().encode("ascii", "ignore").rstrip(" c")
            f_ = f.get_text().encode("ascii", "ignore").rstrip(" c")
            ws = w.get_text().split(" mph")
            ws_ = ws[0]
            if dirs:
                d_ = dirs[i].get_text()
            else:
                d_ = ws[1]
            g_ = g.get_text().rstrip(" mph")
            c_ = c.get_text().rstrip("%")
            h_ = h.get_text().rstrip("%")
            r_ = r.get_text().rstrip(" mm")
            p_ = p.get_text().rstrip(" mb")
            row = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (w_,t_,f_,ws_, d_ ,g_,c_,h_,r_,p_)
            t1 = (i - 1) * 3
            t2 = t1 + 1
            t3 = t1 + 2
            values.append("%s %s:00:00,%s" % (date, utils.format10(t1), row))
            values.append("%s %s:00:00,%s" % (date, utils.format10(t2), row))
            values.append("%s %s:00:00,%s" % (date, utils.format10(t3), row))
        i += 1
    return values


# get url correspond to key
def get_city_url(key, past=True):
    if past:
        keys = {
            "beijing": "https://www.worldweatheronline.com/beijing-weather-history/beijing/cn.aspx",
            "seoul": "https://www.worldweatheronline.com/seoul-weather-history/kr.aspx",
            "daegu": "https://www.worldweatheronline.com/daegu-weather-history/kr.aspx",
            "shenyang": "https://www.worldweatheronline.com/shenyang-weather-history/liaoning/cn.aspx"
        }
    else:
        keys = {
            "seoul" : "https://www.worldweatheronline.com/seoul-weather/kr.aspx"
        }
    return keys[key]


# craw aqi data from source 
def craw_data(key, date):
    data = {
        "__VIEWSTATE": "aVMx4HZm6KLwq3UhFPnPbXmspyROURZzGvRMCeOWPMnPoXIbLT0ibc4Ro1lK13t9YnlhzemPMTH1YRVsoS1fk2ctA1aEEcXM/+/O2NBYDaAwb7sv",
        "__VIEWSTATEGENERATOR": "F960AAB1",
        "ctl00$rblTemp":1,
        "ctl00$rblPrecip":1,
        "ctl00$rblWindSpeed":1,
        "ctl00$rblPressure":1,
        "ctl00$rblVis":2,
        "ctl00$rblheight":1,
        "ctl00$MainContentHolder$txtPastDate": date,
        "ctl00$MainContentHolder$butShowPastWeather":"Get Weather"
    }
    url = get_city_url(key)
    r = requests.post(url, data)
    html = Soup(r.text, "html5lib")
    return html


def craw_future(key, days=1):
    data = {
        "day": days
    }
    url = get_city_url(key, False)
    r = requests.get(url, data)
    html = Soup(r.text, 'html5lib')
    return html


# write data crawled to file
def write_log(filename, output):
    if output:
        with open(filename, "a") as f:
            f.write(output)


def main(args):
    filename = "craw_weather_%s_%s_%s.txt" % (args.city, utils.clear_datetime(args.start), utils.clear_datetime(args.end))
    start = datetime.strptime(args.start, pr.fm)
    if args.end:
        end = datetime.strptime(args.end, pr.fm)
    else:
        end = utils.get_datetime_now()
    start_point = utils.get_datetime_now()
    # output = "timestamp,PM10_VAL,PM2.5_VAL,O3(ppm),NO2(ppm),CO(ppm),SO2(ppm),PM10_AQI,PM2.5_AQI\n"
    output = ""
    length = (end - start).total_seconds() / 86400.0
    save_interval = args.save_interval
    counter = 0
    last_save = 0
    while start <= end:
        now = utils.get_datetime_now()
        if (now - start_point).total_seconds() >= args.interval:
            try:
                counter += 1
                date = "%s-%s-%s" % (start.year, utils.format10(start.month), utils.format10(start.day))
                html = craw_data(args.city, date)
                data = mine_data(date, html)
                if data:
                    output += "\n".join(data) + "\n"
                if (counter - last_save) == save_interval:
                    last_save = counter
                    write_log(filename, output)
                    output = ""
            except Exception as e:
                print(start.strftime(pr.fm), e)
            start = start + timedelta(days=1)
            start_point = now   
            utils.update_progress(counter * 1.0 / length)
    write_log(filename, output)


def get_future(args):
    start_point = utils.get_datetime_now()
    start_point = start_point - timedelta(days=1)
    filename = "data/%s_weather.csv" % args.city
    # use flag to craw first time
    while True:
        now = utils.get_datetime_now()
        if (now - start_point).total_seconds() >= args.interval:
            try:
                # craw 3 days forward
                for i in xrange(4):
                    start_point = start_point + timedelta(days=1)
                    date = "%s-%s-%s" % (start_point.year, utils.format10(start_point.month), utils.format10(start_point.day))
                    html = craw_future(args.city, i)
                    data = mine_data(date, html)
                    if data:
                        output = "\n".join(data) + "\n"
                        write_log(filename, output)
            
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--forward", default=0, type=int, help="get future or get past")
    parser.add_argument("-i", "--interval", default=1, type=int)
    parser.add_argument("-si", "--save_interval", default=48, type=int)
    parser.add_argument("-s", "--start", default="2008-01-01 01:00:00", type=str)
    parser.add_argument("-e", "--end", type=str)
    parser.add_argument("-c", "--city", type=str, default="seoul")
    
    args = parser.parse_args()
    if args.forward:
        get_future(args)
    else:
        main(args)

    
        





        
    
