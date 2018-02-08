# http://mysql-python.sourceforge.net/MySQLdb.html#installation
# https://stackoverflow.com/questions/39149243/how-do-i-connect-to-a-sql-server-database-with-python/46285105#46285105

import MySQLdb as msql 
from datetime import datetime
import time
import argparse
import utils
import properties as p


def crawler(db, t1, t2, url):
    s1 = t1.strftime(p.fm)
    s2 = t2.strftime(p.fm)
    q1 =  ("""select * from sensorParser where gateway_id != "PLNGateway" and timestamp >= "%s" and timestamp < "%s" LIMIT 0, 300000""") % (s1, s2)
    print("Crawling timestamp >= '%s' and timestamp < '%s'" % (s1, s2))
    try:
        db.execute(q1)
        rows = db.fetchall()
        tmp = data_to_csv(rows)
        if not url:
            url = "crawler/%s_sensor.csv" % s2
        else:
            u_ = url.split(".")
            if len(u_) >= 2:
                u_ = u_[:-1]
            url = "".join(u_) + "%s_sensor.csv" % s2
        utils.save_file(url, tmp, False)
    except Exception as e:
        print(e)


def data_to_csv(data):
    tmp = ""
    l = len(data) - 1
    for i, x in enumerate(data):
        l_x = len(x) - 1
        row = ""
        for j, c in enumerate(x):
            if type(c) is datetime:
                row += c.strftime(p.fm)
            elif c is None:
                row += "NULL"
            else:
                row += str(c)
            if j != l_x:
                row += ","
        tmp += row
        if i != l:
            tmp += "\n"
    return tmp


def connect():
    conn = msql.connect(host="220.123.184.109", port=3306, user="adw", passwd="adw2017!", db="kisti")
    return conn, conn.cursor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default=1, type=int)
    parser.add_argument("-u", "--url")
    parser.add_argument("-s", "--start_time")
    parser.add_argument("-e", "--end_time")
    parser.add_argument("-u1", "--url1")
    parser.add_argument("-i", "--interval", type=int, default=3600)

    args = parser.parse_args()
    if args.task == 1:
        ac = datetime(1970,1,1)
        if not args.start_time:
            start = datetime.fromtimestamp(time.time())
        else:
            start = datetime.strptime(args.start_time, p.fm)
        interval = args.interval
        
        try:
            if args.end_time:
                conn, db = connect()
                #2018-01-26 14:20:16
                end = datetime.strptime(args.end_time, p.fm)
                crawler(db, start, end, args.url)
                conn.close()
            else:
                while True:
                    now = datetime.fromtimestamp(time.mktime(time.localtime()))
                    if (now - start).total_seconds() >= interval:
                        conn, db = connect()
                        crawler(db, start, now, args.url)
                        start = now
                        conn.close()
        except Exception as e:
            print(e)
            if conn:
                conn.close()
    elif args.task == 2 and args.url:
        print("==> load file %s" % args.url)
        data = utils.load_file(args.url)
        tmp = data_to_csv(data)
        utils.save_file(args.url1, tmp, False)







