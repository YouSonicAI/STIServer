# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import requests
import json
import time
from mimetypes import guess_type
import os
import sys
from pydantic import BaseModel
from fastapi import Path


def read_in_chunks(filePath, chunk_size=1024*1024*4):
    """
    Lazy function (generator) to read a file piece by piece.
    Default chunk size: 100M
    You can set your own chunk size
    """
    file_object = open(filePath, 'rb')
    cnt = -1
    fileName, suffix = os.path.basename(filePath).split('.')
    print(sys._getframe().f_lineno, fileName, suffix)
    if not os.path.exists(fileName):
        os.mkdir(fileName)
    else:
        print('already exist dir')

    piecePath = os.getcwd() + '\\' + fileName + '\\'
    s = piecePath + fileName + "_"
    print(s)

    #开始分片
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        cnt += 1
        print(cnt)
        tmp = s + str(cnt)
        print(tmp)
        file1 = open(tmp, 'wb')
        file1.write(chunk_data)
        file1.close()

    #逐片上传
    os.chdir(piecePath)
    print(sys._getframe().f_lineno, len(os.listdir()), os.listdir())
    print(sys._getframe().f_lineno, os.path.abspath(os.listdir()[0]))

    for i in range(len(os.listdir())):
        myfile = {'file': open(os.path.abspath(os.listdir()[i]), 'rb')}
        myitem = {
            "identifier": fileName,
            'file_type': suffix,
            'totalNumber': str(len(os.listdir())),
            'currentNum': str(i)
        }
        print(sys._getframe().f_lineno, fileName, suffix, len(os.listdir()), i)
        test_response = requests.post("http://47.99.92.71:8000/prepare", data =json.dumps(myitem))

        if test_response.ok:
            print(f"Upload arguments_{i} successfully!")
            print(test_response.text)
        else:
            print(test_response.text)

        test_response = requests.post("http://47.99.92.71:8000/upload", files = myfile)

        if test_response.ok:
            print(f"Upload piece_{i} successfully!")
            #print(test_response.text)
        else:
            print(test_response.text)

    '''    
    url = 'http://47.99.92.71:8000/results/' + fileName + '.zip'
    r = requests.get(url)
    zipName = os.getcwd() + '\\' + fileName + '.zip'
    with open(zipName, 'wb') as f:
        f.write(r.content)
    '''

url = "http://47.99.92.71:8080/upload"
path = "D:\\zooVideo\\face.mp3"
read_in_chunks(path)
