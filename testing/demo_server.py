from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import urlparse
import tornado.wsgi
import tornado.httpserver

import base64
from io import BytesIO
from PIL import Image

import sys
import os

paths = {}
with open('../path.config', 'r') as f:
    for line in f:
        name, path = line.split(': ')
        print name, path
        paths[name] = path
sys.path.insert(0, paths['pycaffe_root'])
sys.path.insert(0, '/home/jlduan/caffe/python')

import caffe
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2

# ----------------------------------------------------

app = Flask(__name__)

joints = np.arange(31)
Edges = [[0, 1], [1, 2], [2, 3], [3, 4],
         [5, 6], [6, 7], [7, 8], [8, 9],
         [10, 11], [11, 12], [12, 13], [13, 14],
         [15, 16], [16, 17], [17, 18], [18, 19],
         [4, 20], [9, 21], [14, 22], [19, 23],
         [20, 24], [21, 24], [22, 24], [23, 24],
         [24, 25], [24, 26], [24, 27],
         [27, 28], [28, 29], [29, 30]]

J = len(joints)

net = caffe.Net('DeepModel_deploy.prototxt',
                'weights/NYU.caffemodel',
                caffe.TEST)


@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        if request.args.get('id') is None:
            return render_template("index.html")
        else:
            img_id=request.args.get('id');
            filename='{}.jpg'.format(img_id)
            return send_file('./point_clouds/{}'.format(filename),mimetype='image/jpg')

        # return render_template('index.html')
    else:
        fileStream = request.headers['Filedata']
        im = Image.open(BytesIO(base64.b64decode(fileStream))).convert('RGB')
        print im.mode, im.size, im.format
        # im.show()

        # convert to opencv format
        opencv_img = np.array(im)
        opencv_img = opencv_img[:, :, ::-1].copy()

        img_process(opencv_img)

        # return send_file('tmp/example.jpg',mimetype='image/jpg')
        return send_file('tmp/image.jpg', mimetype='image/jpg')


def img_process(img):
    img = cv2.resize(img, (128, 128))
    input = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255. * 2 - 1)
    blobs_in = {'data': input.reshape(1, 1, img.shape[0], img.shape[1])}
    out = net.forward(**blobs_in)  # the input param is a dict
    joint = out['pred'][0]

    x = np.zeros(J)
    y = np.zeros(J)
    z = np.zeros(J)
    cnt=0
    for j in range(J):
        x[j] = joint[joints[j] * 3]
        y[j] = joint[joints[j] * 3 + 1]
        z[j] = joint[joints[j] * 3 + 2]
        cv2.circle(img, (int((x[j] + 1) / 2 * 128), int((-y[j] + 1) / 2 * 128)), 2, (255, 0, 0), 2)

    fig = plt.figure(dpi=300)
    plt.clf()
    ax = fig.add_subplot((111), projection='3d')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.scatter(z, x, y)
    for e in Edges:
        ax.plot(z[e], x[e], y[e], c='b')


        # For axes equal
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([zb], [xb], [yb], 'w')

    plt.tight_layout()
    plt.savefig('./tmp/example.jpg',bbox_inches='tight')
    cv2.imwrite('./tmp/image.jpg',img)

    print img.shape




def start_tornado(_app, port=5800):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(_app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    start_tornado(app, 9000)
