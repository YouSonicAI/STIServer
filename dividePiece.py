import requests
import json
import time
from mimetypes import guess_type
import os
import sys

def read_in_chunks(filePath, chunk_size=1024*1024*100):
    """
    Lazy function (generator) to read a file piece by piece.
    Default chunk size: 100M
    You can set your own chunk size
    """
    file_object = open(filePath, 'rb')
    cnt = -1
    s = os.path.basename(filePath).split('.')[0]
    if not os.path.exists(s):
        os.mkdir(s)
    else:
        print('already exist dir')

    piecePath = os.getcwd() + '\\' + s + '\\'
    s = piecePath + s + "_"
    print(s)

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
