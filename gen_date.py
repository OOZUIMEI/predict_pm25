from datetime import datetime, timedelta
import properties as pr
import utils


start = "2014-01-01 00:00:00" 
end = "2018-01-01 01:00:00"

start_date = datetime.strptime(start, pr.fm)
end_date = datetime.strptime(end, pr.fm)

tmp = ""
while(start_date <= end_date):
    d = datetime.strftime(start_date, pr.fm)
    tmp += "%s\n" % d
    start_date = start_date + timedelta(hours=1)

utils.save_file("checkdate.txt", tmp, use_pickle=False)