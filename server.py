import subprocess
import cherrypy
import time
import json
import numpy as np

import train_sp as engine
from apnet import APNet
import properties as pr
import utils
from  spark_engine import SparkEngine
from train_sp import get_prediction_real_time


class WebApp(object):
    
    @cherrypy.expose
    def index(self):
        return "Hello World"

@cherrypy.expose
class Prediction(object):

    def __init__(self, sparkEngine):
        self.sparkEngine =  sparkEngine
        self.prediction0, self.prediction1 = None, None
        self.avg0, self.avg1 = None, None
        self.timestamp = None
        self.last_time = None
        self.model = APNet(encoder_length=24, decoder_length=24, encode_vector_size=15, batch_size=1, decode_vector_size=9, grid_size=25, forecast_factor=0)

    def predict(self):
        now = utils.get_datetime_now()
        if (not self.prediction0) or not self.last_time or (now - self.last_time).total_seconds() >= 1800:
            self.last_time = now
            preds, timestamp, china = get_prediction_real_time(sparkEngine, self.model)
            self.beijing = china[0,:].flatten().tolist()
            self.shenyang = china[1,:].flatten().tolist()
            # self.prediction0 = (np.array(preds[0]) + 15).tolist()
            # self.prediction1 = (np.array(preds[1]) + 15).tolist()
            self.prediction0 = preds[0]
            self.prediction1 = preds[1]
            self.avg0 = np.mean(self.prediction0, axis=1).tolist()
            self.avg1 = np.mean(self.prediction1, axis=1).tolist()
            self.timestamp = timestamp

    @cherrypy.tools.accept(media="text/plain")
    @cherrypy.expose
    def GET(self):
        self.predict()
        return json.dumps({
            "status": "OK", 
            "data0": self.prediction0, 
            "data1": self.prediction1, 
            "avg0": self.avg0, 
            "avg1": self.avg1, 
            "timestamp": self.timestamp,
            "beijing": self.beijing,
            "shenyang": self.shenyang
        })

@cherrypy.expose
class Time(object):
    
    @cherrypy.tools.accept(media="text/plain")
    @cherrypy.expose
    def GET(self):
        date = utils.get_datetime_now()
        return json.dumps({"datetime": date.strftime(pr.fm)})


if __name__ == "__main__":
    subprocess.call("source activate tensorflow", shell=True)
    
    cherrypy.config.update({
        'server.socket_host': pr.host,
        'server.socket_port': pr.port
    })

    conf = {
        '/prediction': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-type', 'application/json'), ('Access-Control-Allow-Origin', '*')]
        },
        '/time': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-type', 'application/json'), ('Access-Control-Allow-Origin', '*')]
        }
    }
    sparkEngine = SparkEngine()
    app = WebApp()
    app.prediction = Prediction(sparkEngine)
    app.time = Time()
    cherrypy.quickstart(app, '/', conf)
    
