import subprocess
import cherrypy
import json
import numpy as np

import train_sp as engine
import properties as pr
from  spark_engine import SparkEngine
from train_sp import get_prediction_real_time, get_districts_preds


class WebApp(object):
    
    @cherrypy.expose
    def index(self):
        return "Hello World"

@cherrypy.expose
class Prediction(object):

    def __init__(self, sparkEngine):
        self.sparkEngine =  sparkEngine
        self.prediction = None
        self.avg = None

    @cherrypy.tools.accept(media="text/plain")
    @cherrypy.expose
    def GET(self):
        if not self.prediction:
            preds = get_prediction_real_time(sparkEngine)
            self.prediction = get_districts_preds(preds)
            self.avg = np.mean(self.prediction, axis=1).tolist()
        return json.dumps({"status": "OK", "data": self.prediction, "avg": self.avg})


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
    