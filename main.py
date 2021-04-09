import youtube_dl
import os
import cv2 as cv
import ffmpeg
import numpy as np
import json
import http.server
import socketserver
import webbrowser

colorDivs = 5
fps = 10
slowDownFactor = 4
port = 42069

youtubeUrl = input('Give a youtube link: ').strip()
#youtubeUrl = 'https://www.youtube.com/watch?v=-5I24Hr0sWY'
if os.path.exists('video.mp4'):
    os.remove('video.mp4')

if os.path.exists('lowFPS.mp4'):
    os.remove('lowFPS.mp4')
ydl_opts = {
    'outtmpl': 'video.mp4',
    'format': 'mp4'
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtubeUrl])

ffmpeg.input('video.mp4').video.filter('fps', fps=fps).trim(start=0, end=10).output('lowFPS.mp4').run()

videoInput = cv.VideoCapture('lowFPS.mp4')
commandOutput = {
    'frames': []
}
frameSize = []
while videoInput.isOpened():
    ret, frame = videoInput.read()
    if not ret:
        break
    print("frame")
    frame = cv.flip(frame, 0)
    frameSize = frame.shape[:-1]
    ret, labels, colors = cv.kmeans(np.float32(frame.reshape((-1, 3))),
                                    colorDivs,
                                    None,
                                    (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    10,
                                    cv.KMEANS_PP_CENTERS)
    labels = labels.reshape(frame.shape[:-1])
    contourList = []
    for i, color in enumerate(colors):
        singleChannel = cv.inRange(labels, i, i)
        contours, hierarchy = cv.findContours(singleChannel, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        contours = [a for a in contours if cv.contourArea(a) != np.prod(frame.shape[:-1]) and len(a) > 2 and cv.contourArea(a) > np.prod(frame.shape[:-1])*len(contours)*colorDivs/80000]
        contourList.append([contours, np.floor(color).astype(np.uint8)])
    frameCommands = []
    for i, contourGroup in enumerate(contourList):
        for contour in contourGroup[0]:
            contour = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
            contour = [tuple(point.tolist()[0]) for point in contour]
            out = {'latex': '\\operatorname{polygon}' + '({})'.format(contour),
                   'color': "#{0:02x}{1:02x}{2:02x}".format(contourGroup[1][2], contourGroup[1][1], contourGroup[1][0]),
                   'fillOpacity': 1}
            frameCommands.append(out)
    commandOutput['frames'].append(frameCommands)
videoInput.release()

scriptString = """var elt = document.getElementById('calculator');
var calculator = Desmos.GraphingCalculator(elt);
calculator.setMathBounds({{left:0, bottom: 0, right: {}, top: {}}});
var commandObj = {};
var frameInterval = {};
for(var i = 0; i < commandObj.frames.length; i++){{
setTimeout(setNClear, frameInterval*i, commandObj.frames[i]);
}}
function setNClear(frames){{
calculator.removeExpressions(calculator.getExpressions());
calculator.setExpressions(frames);
}}""".format(frameSize[1], frameSize[0], json.dumps(commandOutput), (1000 // fps)*slowDownFactor)

with open("main.js", "w") as f_output:
    f_output.write(scriptString)


handler = http.server.SimpleHTTPRequestHandler
print("Server Up")
webbrowser.open('http://127.0.0.1:{}/fullscreen.html'.format(port))
with socketserver.TCPServer(("", port), handler) as httpd:
    httpd.serve_forever()