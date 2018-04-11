# _*_ coding: utf-8 _*_
from bs4 import BeautifulSoup as Soup
import requests
from argparse import ArgumentParser
from datetime import datetime, timedelta
import utils
import properties as pr


# clear out html and get data
def mine_data(html):
    tables = html.find('table', attrs={"class": "tbl2"})
    avg = tables.find('tr', attrs={"class": "ft_b ft_point8"})
    values = []
    if avg:
        td = avg.find_all('td')
        for v in td:
            txt = v.get_text()
            if "-" not in txt:
                values.append(txt.encode('ascii', 'ignore'))
        values = [float(x) for x in values[1:7]]
        values.append(utils.AQIPM10(values[0]))
        values.append(utils.AQIPM25(values[1]))
    return values


def mine_all_area(html):
    tables = html.find('table', attrs={"class": "tbl2"})
    body = tables.find('tbody')
    all_values = []
    if body:
        all_tr = body.find_all("tr")
        if all_tr:
            for ku, r in enumerate(all_tr):
                td = r.find_all('td')
                values = []
                for i, v in enumerate(td):
                    txt = v.get_text()
                    txt = "".join(txt.replace("\n", "").split(" "))
                    if i > 0 and i < 7 and "-" not in txt:
                        txt = txt.encode('ascii', 'ignore')
                        if txt:
                            values.append(float(txt))
                        else:
                            values.append(all_values[0][i])
                    elif i == 0:                
                        values.append(ku)
                values.append(utils.AQIPM10(values[1]))
                values.append(utils.AQIPM25(values[2]))
                all_values.append(values)
    return all_values


# format hour, day < 10 to 10 format
def format10(no):
    if no < 10:
        return "0" + str(no)
    else:
        return str(no)


# craw aqi data from source 
def craw_data(year, month, date, hour):
    data = {
        "bodyCommonMethod": "measure",
        "bodyCommonHtml": "air_city.htm",
        "msrntwCode": "A",
        "grp1": "pm25",
        "pNum": "1",
        "scrollTopVal": "713.6363525390625",
        "lGroup": "1",
        "mGroup":"",
        "sGroup": "TIME",
        "cal2": "%s" % year,
        "cal2Month": "%s" % month,
        "cal2Day": "%s" % date,
        "time": "%s" % hour,
        "x": "21",
        "y": "32"
    }
    r = requests.post("http://cleanair.seoul.go.kr/air_city.htm?method=measure&citySection=CITY", data)
    html = Soup(r.text, "html5lib")
    return html
    

# write data crawled to file
def write_log(filename, output):
    if output:
        with open(filename, "a") as f:
            f.write(output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--test", default=1, type=int)
    parser.add_argument("-i", "--interval", default=5, type=int)
    parser.add_argument("-si", "--save_interval", default=48, type=int)
    parser.add_argument("-s", "--start", default="2008-01-01 01:00:00", type=str)
    parser.add_argument("-e", "--end", type=str)
    parser.add_argument("-a", "--get_all", type=int, default=0)
    
    args = parser.parse_args()
    if args.test:
        with open("test_seoul_aqi.html") as file:
            html = Soup(file.read(), "html5lib")
            # timestamp,K,PM-10(㎍/㎥),PM-2.5(㎍/㎥),O3(ppm),NO2(ppm),CO(ppm),SO2(ppm)
            # mine_data(html)
            values = mine_all_area(html)
            timestamp = "2008-01-01 01:00:00"
            output = ""
            for x in values:
                output += timestamp + "," + utils.array_to_str(x, ",") + "\n"
    else:
        filename = "craw_seoul_aqi_%s_%s.txt" % (utils.clear_datetime(args.start), utils.clear_datetime(args.end))
        start = datetime.strptime(args.start, pr.fm)
        if args.end:
            end = datetime.strptime(args.end, pr.fm)
        else:
            end = utils.get_datetime_now()
        start_point = utils.get_datetime_now()
        # output = "timestamp,PM10_VAL,PM2.5_VAL,O3(ppm),NO2(ppm),CO(ppm),SO2(ppm),PM10_AQI,PM2.5_AQI\n"
        output = ""
        length = (end - start).total_seconds() / 3600.0
        save_interval = args.save_interval
        counter = 0
        last_save = 0
        while start <= end:
            now = utils.get_datetime_now()
            if (now - start_point).total_seconds() >= args.interval:
                hour = start.hour
                tmp = start
                if tmp.hour == 0:
                    tmp = tmp - timedelta(hours=1)
                    hour = "24"
                else:
                    hour = format10(tmp.hour)
                year = tmp.year
                month = format10(tmp.month)
                date = format10(tmp.day)
                try:
                    counter += 1
                    html = craw_data(year, month, date, hour)
                    # data = mine_data(html)
                    if  args.get_all:
                        data = mine_all_area(html)
                        for x in data:
                            output += start.strftime(pr.fm) + "," + utils.array_to_str(x, ",") + "\n"
                            if (counter - last_save) == save_interval:
                                last_save = counter
                                write_log(filename, output)
                                output = ""
                    else:
                        data = mine_data(html)
                        if data:
                            output += start.strftime(pr.fm) + "," + utils.array_to_str(data, ",") + "\n"
                        if (counter - last_save) == save_interval:
                            last_save = counter
                            write_log(filename, output)
                            output = ""
                except Exception as e:
                    print(start.strftime(pr.fm), e)
                start = start + timedelta(hours=1)
                start_point = now   
                utils.update_progress(counter * 1.0 / length)
        write_log(filename, output)
        





        
    
