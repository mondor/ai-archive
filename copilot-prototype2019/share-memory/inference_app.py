from flask import Flask
from flask import request
from flask import jsonify
from flask import json
from flask import Response
from multiprocessing import Process
import time
import os

app = Flask(__name__)

def info(title):
    print "### " + title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):  # only available on Unix
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()


def expensive_function():
    i=0
    while i < 10:
        i = i+1
        #print "loop running %d" % i
        time.sleep(1)

@app.route("/<training_job_id>", methods=["GET"])
def inference(training_job_id):
    info('inference')
    global p
    p = Process(target=expensive_function)
    p.start()
    return Response(
        response=json.dumps({"Okay": training_job_id}),
        status=200,
        mimetype="application/json")


if __name__ == "__main__":
    info('main line')
    app.run(debug=False)