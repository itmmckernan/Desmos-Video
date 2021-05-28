import youtube_dl
import os
import cv2 as cv
import ffmpeg
import numpy as np
import json
import http.server
import socketserver
import webbrowser
import asyncio


colorDivs = 5
fps = 5
slowDownFactor = 4
port = 42069

magicNumber = '\x68\x74\x74\x70\x73\x3a\x2f\x2f\x77\x77\x77\x2e\x79\x6f\x75\x74\x75\x62\x65\x2e\x63\x6f\x6d\x2f\x77' \
              '\x61\x74\x63\x68\x3f\x76\x3d\x54\x74\x37\x62\x7a\x78\x75\x72\x4a\x31\x49'

if os.path.exists('video.mp4'):
    os.remove('video.mp4')

if os.path.exists('lowFPS.mp4'):
    os.remove('lowFPS.mp4')
ydl_opts = {
    'outtmpl': 'video.mp4',
    'format': 'mp4'
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([magicNumber])

ffmpeg.input('video.mp4').video.filter('fps', fps=fps).trim(start=0, end=15).output('lowFPS.mp4').run()

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
        contours = [a for a in contours if cv.contourArea(a) != np.prod(frame.shape[:-1]) and len(a) > 2 and cv.contourArea(a) > np.prod(frame.shape[:-1])*len(contours)*colorDivs/120000]
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
    frameCommands = sorted(frameCommands, key=len, reverse=True)
    commandOutput['frames'].append(frameCommands)
videoInput.release()