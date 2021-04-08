import youtube_dl
import os
import cv2 as cv
import ffmpeg
import time
import numpy as np


colorDivs = 4
#youtubeUrl = input('Give a youtube link: ').strip()
youtubeUrl = 'https://www.youtube.com/watch?v=MVMCBsqi0L8'
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

ffmpeg.input('video.mp4').video.filter('fps', fps=5).output('lowFPS.mp4').run()


videoInput = cv.VideoCapture('lowFPS.mp4')
while videoInput.isOpened():
    ret, frame = videoInput.read()
    if not ret:
        break
    cv.imshow('Frame1', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    stepValue = 256/colorDivs
    contourList = []
    for colorFrame in cv.split(frame):
        for i in range(colorDivs):
            ret, segImage = cv.threshold(colorFrame, stepValue * i,  stepValue * (i + 1)-1, cv.THRESH_BINARY)
            contours, hierarchy = cv.findContours(segImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contourList.extend(contours)

    #this is not the most elegant way to do this, but to be honest i dont give a shit
    #update: i give a shit
    contourOutput = []
    contourList = [cv.convexHull(x) for x in contourList]
    contourList = [x for x in contourList if cv.contourArea(x) > 1000]
    cv.imshow('Framzsae', cv.drawContours(frame, contourList, -1, (255, 0, 255), 1))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    while len(contourList) > 0:
        mainContour = contourList.pop(0)
        mainContourArea = cv.contourArea(mainContour)
        contourOutput.append(mainContour)
        indexesToDel = []
        for i, contourToMatch in enumerate(contourList):
            matchingImage = np.zeros(frame.shape[:-1], np.uint8)
            matchingImage = cv.drawContours(matchingImage, mainContour, -1, [120], thickness=-1)
            matchingImage = cv.drawContours(matchingImage, contourToMatch, -1,  [120], thickness=-1)
            overlapArea = np.sum(np.greater(matchingImage, 230))
            cv.imshow('test', matchingImage)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            print(overlapArea, mainContourArea)
            if overlapArea/mainContourArea > 0:
                indexesToDel.append(i)
        np.delete(contourList, indexesToDel)
    cv.imshow('Frame', cv.drawContours(frame, contourOutput, -1, (128, 0, 0), 3))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

videoInput.release()