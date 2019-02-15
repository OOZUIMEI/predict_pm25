import math 


# convert pm10 micro value to aqi value
def AQIPM10(Concentration):
    Conc = float(Concentration)
    c = math.floor(Conc)
    if (c >= 0 and c < 55):
        AQI = Linear(50, 0, 54, 0, c)
    elif(c >= 55 and c < 155):
        AQI = Linear(100, 51, 154, 55, c)
    elif(c >= 155 and c < 255):
        AQI = Linear(150, 101, 254, 155, c)
    elif(c >= 255 and c < 355):
        AQI = Linear(200, 151, 354, 255, c)
    elif(c >= 355 and c < 425):
        AQI = Linear(300, 201, 424, 355, c)
    elif(c >= 425 and c < 505):
        AQI = Linear(400, 301, 504, 425, c)
    elif(c >= 505 and c < 605):
        AQI = Linear(500, 401, 604, 505, c)
    else:
        AQI = 0
    return AQI


# convert pm25 micro value to aqi value
def AQIPM25(Concentration):
    Conc = float(Concentration)
    c = (math.floor(10 * Conc)) / 10
    if (c >= 0 and c < 12.1):
        AQI = Linear(50, 0, 12, 0, c)
    elif (c >= 12.1 and c < 35.5):
        AQI = Linear(100, 51, 35.4, 12.1, c)
    elif (c >= 35.5 and c < 55.5):
        AQI = Linear(150, 101, 55.4, 35.5, c)
    elif (c >= 55.5 and c < 150.5):
        AQI = Linear(200, 151, 150.4, 55.5, c)
    elif (c >= 150.5 and c < 250.5):
        AQI = Linear(300, 201, 250.4, 150.5, c)
    elif (c >= 250.5 and c < 350.5):
        AQI = Linear(400, 301, 350.4, 250.5, c)
    elif (c >= 350.5 and c < 500.5):
        AQI = Linear(500, 401, 500.4, 350.5, c)
    else:
        AQI = 0
    return AQI


def InvLinear(AQIhigh, AQIlow, Conchigh, Conclow, a):
    c=((a-AQIlow)/(AQIhigh-AQIlow))*(Conchigh-Conclow)+Conclow
    return c


# aqi to concentration
def ConcPM25(a):
    if a>=0 and a<=50:
        ConcCalc=InvLinear(50,0,12,0,a)
    elif a>50 and a<=100:
        ConcCalc=InvLinear(100,51,35.4,12.1,a)
    elif a>100 and a<=150:
        ConcCalc=InvLinear(150,101,55.4,35.5,a)
    elif a>150 and a<=200:
        ConcCalc=InvLinear(200,151,150.4,55.5,a)
    elif a>200 and a<=300:
        ConcCalc=InvLinear(300,201,250.4,150.5,a)
    elif a>300 and a<=400:
        ConcCalc=InvLinear(400,301,350.4,250.5,a)
    elif a>400 and a<=500:
        ConcCalc=InvLinear(500,401,500.4,350.5,a)
    else:
        ConcCalc=0
    return ConcCalc


# aqi to concentration
def ConcPM10(a):
    if a>=0 and a<=50:
        ConcCalc=InvLinear(50,0,54,0,a)
    elif a>50 and a<=100:
        ConcCalc=InvLinear(100,51,154,55,a)
    elif a>100 and a<=150:
        ConcCalc=InvLinear(150,101,254,155,a)
    elif a>150 and a<=200:
        ConcCalc=InvLinear(200,151,354,255,a)
    elif a>200 and a<=300:
        ConcCalc=InvLinear(300,201,424,355,a)
    elif a>300 and a<=400:
        ConcCalc=InvLinear(400,301,504,425,a)
    elif a>400 and a<=500:
        ConcCalc=InvLinear(500,401,604,505,a)
    else:
        ConcCalc = 0.
    return ConcCalc


def Linear(AQIhigh, AQIlow, Conchigh, Conclow, Concentration):
    Conc = float(Concentration)
    a = ((Conc - Conclow) / (Conchigh - Conclow)) * (AQIhigh - AQIlow) + AQIlow
    # linear = round(a)
    return a