import sys, time
import numpy as np 
import math


prefix = "/home/alex/Documents/datasets/sensorParser.csv"
valid_data = "/home/alex/Documents/datasets/sensor_data.csv"
length = 27050177;

cols = [3,40,41,31,33,22,26,29,10,14,18,27,34,38,39]
pm25range = [(0, 50),(51,100),(101,150),(151,200),(201,300),(301,400),(401,500)]
pm25text = ['Good','Moderate','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous','Hazardous']


def update_progrss(progress, sleep=0.01, barlength = 20):
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error"
	if progress < 0:
		progress = 0
		status = "halt"
	if progress >= 1.0:
		status = "done"
	block = int(round(barlength*progress))
	text = "\rpercent: [{0}] {1}% {2}".format("#"*block + "-"*(barlength - block), progress * 100, status)
	sys.stdout.write(text)
	sys.stdout.flush()
	time.sleep(sleep)


def get_label(pm25):
	i = 0
	if pm25 <= 200:
		i = math.ceil(float(pm25) / 50) - 1
	else:
		pm25 -= 200
		i = 3 + math.ceil(float(pm25) / 100)
	return int(i)


def get_label2(pm25):
	i, j = 0, 0
	for x, y in pm25range:
		if pm25 >= x and pm25 <= y:
			i = j
			break
		j += 1
	return i



def update_label(url, length):
	# "2017-06-08 22:12:10","NULL","NULL","26.7","84","18","21","NULL","NULL","NULL","NULL","457","1013.41","NULL","NULL
	with open(url) as file:
		total = 0
		end = length - 1
		tmp_l = ""
		for line in file:
			line = line.rstrip("\n")
			tmp = 
			if total < end:
				tmp = tmp + '\n'
			tmp_l += tmp
			if total and total % 100000 == 0:
				print("write part %i" % (total / 100000))
				tmp_l2 = tmp_l
				tmp_l = ""
				with open(valid_data, 'a') as wf:
					wf.write(tmp_l2)
			total += 1
			update_progrss(float(total) / length)
		print("write part last part")
		with open(valid_data, 'a') as wf:
			wf.write(tmp_l)



if __name__ == "__main__":

	with open(prefix) as file:
		total = 0
		end = length - 1
		tmp_l = ""
		for line in file:
			line = line.rstrip("\n")
			data = line.split(",")
			data = np.array(data)
			data = data[cols]
			if total > 0:
				p = int(float(data[12].replace("\"", "")))
				if p >= 10000:
					p_ = len(str(p)) -  4
					if p_ > 0:
						p = float(p) / (pow(10, p_))
						data[12] = "\"" + str(p) + "\"" 
			tmp = ','.join([x for x in data]) 
			if total < end:
				tmp = tmp + '\n'
			tmp_l += tmp
			if total and total % 100000 == 0:
				print("write part %i" % (total / 100000))
				tmp_l2 = tmp_l
				tmp_l = ""
				with open(valid_data, 'a') as wf:
					wf.write(tmp_l2)
			total += 1
			update_progrss(float(total) / length)
		print("write part last part")
		with open(valid_data, 'a') as wf:
			wf.write(tmp_l)