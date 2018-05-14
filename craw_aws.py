# _*_ coding: utf-8 _*_
from lxml import etree
import requests
from argparse import ArgumentParser
from datetime import datetime, timedelta
import time
import utils
import properties as pr


def mine_data(html):
    all_values = []
    doc = etree.HTML(html)
    # records
    path1 = doc.xpath("//table[@id='TRptList']/tbody/tr")
    # time
    path2 = doc.xpath("//table[@id='TRptFixed']/tbody/tr")
    
    for r, t in zip(path1, path2):
        record = []
        t_td = t.xpath("./td/text()")
        if  t_td: 
            record.append(t_td[0].encode('ascii', 'ignore'))
            for td in r.xpath("./td/text()"):
                record.append(td.encode('ascii', 'ignore'))
        all_values.append(record)
    # agg = html.find("div", attrs={"id", "RptFooter"}).find("tr").find_all("td")
    # agg_record = []
    # print(all_values)
    return all_values


# format hour, day < 10 to 10 format
def format10(no):
    if no < 10:
        return "0" + str(no)
    else:
        return str(no)


# craw aqi data from source 
def craw_data(timestamp, area):
    data = {
        "EXCEL": "SCREEN",
        "MODE": "SEARCH",
        "SDATAKEY":" ",
        "VIEW_MIN":" ",
        "VIEW_MAX":" ",
        "VIEW_SITE":" ",
        "VIEW_WIND":" ",
        "AWS_ID": area,
        "RptSDATE": timestamp,
        "RptSH": "00",
        "RptSM": "01",
        "RptEH": "24",
        "RptEM": "00"
    }
    r = requests.post("http://aws.seoul.go.kr/Report/RptWeatherMinute.asp", data)
    # html = Soup(r.text, "html5lib")
    return r.text


def download(year, month, date, area):
    timestamp = "%s-%s-%s" % (year, month, date)
    data = {
        "EXCEL": "EXCEL",
        "MODE": "SEARCH",
        "SDATAKEY":"",
        "VIEW_MIN":"",
        "VIEW_MAX":"",
        "VIEW_SITE":"",
        "VIEW_WIND":"",
        "AWS_ID": area,
        "RptSDATE": timestamp,
        "RptSH": "00",
        "RptSM": "01",
        "RptEH": "24",
        "RptEM": "00"
    }
    r = requests.post("http://aws.seoul.go.kr/Report/RptWeatherMinute.asp", data, stream=True)
    with open(timestamp + "_" + str(area) , 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)


# perform craw in loop or by files
def craw_data_controller(output, counter, last_save, save_interval, tmp):
    year = tmp.year
    month = format10(tmp.month)
    date = format10(tmp.day)
    timestamp = "%s-%s-%s" % (year, month, date)
    counter += 1
    try:
        for dis in pr.district_codes:
            html = craw_data(timestamp, dis)
            values = mine_data(html)
            for x in values:
                output += timestamp + " " + x[0] + ":00" + utils.array_to_str(x[1:], ",") + "\n"
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--test", default=1, type=int)
    parser.add_argument("-i", "--interval", default=5, type=int)
    parser.add_argument("-f", "--file", default="", type=str)
    parser.add_argument("-si", "--save_interval", default=48, type=int)
    parser.add_argument("-s", "--start", default="2009-03-01 00:00:00", type=str)
    parser.add_argument("-e", "--end", type=str)
    
    args = parser.parse_args()
    save_interval = args.save_interval
    if args.test:
        with open("test_aws.html") as file:
            values = mine_data(file.read())
            output = ""
            for x in values:
                output += utils.array_to_str(x, ",") + "\n"
            write_log("test_aws_.txt", output)
        # download("2018", "05", "01", 1158)
    else:
        filename = "craw_aws_%s_%s.txt" % (utils.clear_datetime(args.start), utils.clear_datetime(args.end))
        start = datetime.strptime(args.start, pr.fm)
        if args.end:
            end = datetime.strptime(args.end, pr.fm)
        else:
            end = utils.get_datetime_now()
        start_point = utils.get_datetime_now()
        # output = "timestamp,PM10_VAL,PM2.5_VAL,O3(ppm),NO2(ppm),CO(ppm),SO2(ppm),PM10_AQI,PM2.5_AQI\n"
        output = ""
        length = (end - start).total_seconds() / 86400
        counter = 0
        last_save = 0
        while start <= end:
            now = utils.get_datetime_now()
            if (now - start_point).total_seconds() >= args.interval:
                tmp = start
                output, counter, last_save = craw_data_controller(output, counter, last_save, save_interval, tmp)
                start = start + timedelta(days=1)
                start_point = now   
                utils.update_progress(counter * 1.0 / length)
        write_log(filename, output)