import subprocess
import cherrypy
import json
import train_sp as engine
import properties as pr


class WebApp(object):
    
    @cherrypy.expose
    def index(self):
        return "Hello World"

@cherrypy.expose
class Prediction(object):

    @cherrypy.tools.accept(media="text/plain")
    @cherrypy.expose
    def GET(self):
        engine.predict_real_time()
        return json.dumps({"status": "OK"})


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

    app = WebApp()
    app.prediction = Prediction()
    cherrypy.quickstart(app, '/', conf)