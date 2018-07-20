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
                        txt = txt.rstrip("\n")   
                        index = pr.districts.index(txt)
                        values.append(index)
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


# perform craw in loop or by files
def craw_data_controller(output, filename, counter, last_save, save_interval, tmp, hour, timestamp):
    year = tmp.year
    month = format10(tmp.month)
    date = format10(tmp.day)
    counter += 1
    try:
        html = craw_data(year, month, date, hour)
        # data = mine_data(html)
        data = mine_all_area(html)
        for x in data:
            output += timestamp + "," + utils.array_to_str(x, ",") + "\n"
            if (counter - last_save) == save_interval:
                last_save = counter
                write_log(filename, output)
                output = ""
    except Exception as e:
        print(e)
    return output, counter, last_save


# write data crawled to file
def write_log(filename, output):
    if output:
        with open(filename, "a") as f:
            f.write(output)


def main(args, cont=False):
    save_interval = args.save_interval
    start = datetime.strptime(args.start, pr.fm)
    filename = "data/seoul_aqi.csv"
    start_point = utils.get_datetime_now()
    output = ""
    counter = 0
    last_save = 0
    if not cont:
        cond = start <= end
        if args.end:
            end = datetime.strptime(args.end, pr.fm)
        else:
            end = utils.get_datetime_now()
        length = (end - start).total_seconds() / 86400
    else:
        cond = True
    while cond:
        now = utils.get_datetime_now()
        if (now - start_point).total_seconds() >= args.interval:
            start_point = now
            if (now - start).total_seconds() > 3600:
                hour = start.hour
                tmp = start
                if tmp.hour == 0:
                    tmp = tmp - timedelta(hours=1)
                    hour = "24"
                else:
                    hour = format10(tmp.hour)
                st_ = start.strftime(pr.fm)
                output, counter, last_save = craw_data_controller(output, filename, counter, last_save, save_interval, tmp, hour, st_)
                # move pointer for timestep
                start = start + timedelta(hours=1)
                if not cont:
                    utils.update_progress(counter * 1.0 / length)
                else:
                    write_log(filename, output)
                    output = ""
                # else:
                #     print("Crawling %s" % st_)
    write_log(filename, output)      


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--forward", default=1, type=int, help="continuously collecting")
    parser.add_argument("-i", "--interval", default=5, type=int, help="secondly interval")
    parser.add_argument("-si", "--save_interval", default=48, type=int, help="secondly saving interval")
    parser.add_argument("-s", "--start", default="2008-01-01 01:00:00", type=str)
    parser.add_argument("-e", "--end", type=str)
    
    args = parser.parse_args()
    main(args, args.forward)