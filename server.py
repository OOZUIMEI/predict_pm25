import subprocess
import cherrypy
import json
import numpy as np

import train_sp as engine
import properties as pr
import utils
from  spark_engine import SparkEngine
from train_sp import get_prediction_real_time, get_districts_preds, aggregate_predictions


class WebApp(object):
    
    @cherrypy.expose
    def index(self):
        return "Hello World"

@cherrypy.expose
class Prediction(object):

    def __init__(self, sparkEngine):
        self.sparkEngine =  sparkEngine
        self.prediction = None
        self.timestamp = None
        self.avg = None
        self.last_time = None

    @cherrypy.tools.accept(media="text/plain")
    @cherrypy.expose
    def GET(self):
        now = utils.get_datetime_now()
        if not self.prediction or not self.last_time or (now - self.last_time).total_seconds() >= 1800:
            self.last_time = now
            preds, timestamp = get_prediction_real_time(sparkEngine)
            self.prediction = aggregate_predictions(preds)
            self.avg = np.mean(self.prediction, axis=1).tolist()
            self.timestamp = timestamp
        return json.dumps({"status": "OK", "data": self.prediction, "avg": self.avg, "timestamp": self.timestamp})


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
        }
    }
    sparkEngine = SparkEngine()
    app = WebApp()
    app.prediction = Prediction(sparkEngine)
    cherrypy.quickstart(app, '/', conf)
    
