import base64
import os
import sys
import re
import shutil
import stat
from email.utils import formatdate
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import quote

import aiofiles
from fastapi import Body, FastAPI, File, Path as F_Path, Request, UploadFile
from starlette.responses import StreamingResponse

import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from model.FpnAttentionEncoder import AttentionFPN
from dataloader import DatasetSlice, vad_collate_fn
from torch.utils.data import DataLoader
from moviepy.editor import AudioFileClip
from scipy.io.wavfile import write
from tqdm import tqdm
import argparse
import torch
import time
import onnxruntime as ort
import numpy as np
import uvicorn

from typing import Optional
import zipfile
import io
from fastapi import FastAPI, Response
from pydantic import BaseModel
from starlette.responses import FileResponse


onnx_path = "/root/Project/Model/ImageClassifier2.onnx"
overlap = 1
batch_size = 4
average_iterval = 5
ort_session = ort.InferenceSession(onnx_path)

app = FastAPI(docs_url="/docs")

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_file_path = Path(base_dir, './uploads')
result_file_path = Path(base_dir, './results')


myIdentifier = str('nihao')
myFileType = str('mp4')
myTotalNumber = str(0)
myCurrentNum = str(0)
myZipFile = str('')


@app.get("/results")
async def getUrl():
    global myZipFile
    return  myZipFile

@app.post("/prepare")
async def update_value(
        identifier: str = Body(..., description="文件名称（不含后缀）"),
        file_type: str = Body(..., description="文件类型/后缀"),
        totalNumber: str = Body(..., description="文件总片数"),
        currentNum: str = Body(..., description="文件分片序号（初值为0）")
        ):
    global myIdentifier
    global myFileType
    global myTotalNumber
    global myCurrentNum
    myIdentifier = identifier
    myFileType = file_type
    myTotalNumber = totalNumber
    myCurrentNum = currentNum
    print(sys._getframe().f_lineno, myIdentifier, myFileType, myTotalNumber, myCurrentNum)


@app.post("/upload")
async def upload_file(
    #request: Request,
    #identifier: str = Body(..., description="文件名称（不含后缀）"),
    #file_type: str = Body(..., description="文件类型/后缀"),
    #totalNumber: str = Body(..., description="文件总片数"),
    #currentNum: str = Body(..., description="文件分片序号（初值为0）"),
    file: UploadFile = File(..., description="文件")
):

    """文件分片上传"""
    print(sys._getframe().f_lineno, myIdentifier, myFileType, myTotalNumber, myCurrentNum)
    path = Path(upload_file_path, myIdentifier)
    print(sys._getframe().f_lineno, path)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = Path(path, f'{myIdentifier}_{myCurrentNum}')
    if not os.path.exists(file_name):
        async with aiofiles.open(file_name, 'wb') as f:
            await f.write(await file.read())
    
    print(sys._getframe().f_lineno, f"upload piece_{myCurrentNum} over")

    if int(myCurrentNum) + 1 != int(myTotalNumber):
        return {
                'code': 1,
                'chunk': f'{myIdentifier}_{myCurrentNum}'
                }
    else:
        """合并分片文件"""
        target_file_name = Path(upload_file_path, f'{myIdentifier}.{myFileType}')

        try:
            async with aiofiles.open(target_file_name, 'wb+') as target_file:  # 打开目标文件
                for i in range(len(os.listdir(path))):
                    temp_file_name = Path(path, f'{myIdentifier}_{i}')
                    async with aiofiles.open(temp_file_name, 'rb') as temp_file:  # 按序打开每个分片
                        data = await temp_file.read()
                        await target_file.write(data)  # 分片内容写入目标文件
        except Exception as e:
            print(sys._getframe().f_lineno, "merged failed")
            return {
                    'code': 0,
                    'error': f'合并失败：{e}'
                    }
        shutil.rmtree(path)  # 删除临时目录

    print(sys._getframe().f_lineno, "merged successfully")
    
    """深度学习处理文件"""
    process_file_name = str(target_file_name)
    print(sys._getframe().f_lineno, process_file_name)
    run_model(process_file_name, result_file_path)
    f1 = Path(result_file_path, myIdentifier + '.csv')
    f2 = Path(result_file_path, myIdentifier + '.png')
    f3 = Path(result_file_path, myIdentifier + '.wav')
    li = [f1, f2, f3]
    print(li)
    return zipfiles(myIdentifier, li)
    
    '''
    return {
        'code': 1,
        'filename': f'{myIdentifier}.{myFileType}'
    }
    '''


def get_file_byte(filename):  # filename可以是文件，也可以是压缩包
    with open(filename, "rb") as f:
        while True:
            content = f.read(1024)
            if content:
                yield content
            else:
                break


def zipfiles(name, resultFileNames):
    zip_filename = name + ".zip"
    
    zipFilePath = str(Path(result_file_path,zip_filename))
    global myZipFile
    myZipFile = zipFilePath
    print(sys._getframe().f_lineno, zipFilePath)
    #s = io.BytesIO()
    zf = zipfile.ZipFile(zipFilePath, "w")

    for fpath in resultFileNames:
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)
        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    #resp = Response(zipFilePath.getvalue(), media_type="application/x-zip-compressed", headers={
    #    'Content-Disposition': f'attachment;filename={zip_filename}'
    #})
    #return resp
 
    response = StreamingResponse(get_file_byte(zipFilePath))
    print(sys._getframe().f_lineno)
    return response
    #return zipFilePath
    #return FileResponse(zipFilePath, zip_filename)



@app.put("/process/{file_name}")
async def process_audio(request: Request, file_name: str = F_Path(..., description="文件名称（含后缀）")):
    print(sys._getframe().f_lineno, file_name)
    process_file_name = str(Path(upload_file_path, file_name))
    name, *suffix = file_name.rsplit('.', 1)
    print(sys._getframe().f_lineno, process_file_name, name, suffix[0])
    run_model(process_file_name, result_file_path)
    f1 = Path(result_file_path, name + '.csv')
    f2 = Path(result_file_path, name + '.png')
    f3 = Path(result_file_path, name + '.wav')
    li = [f1, f2, f3]
    print(li)
    return zipfiles(name, li)


'''    
@app.get("/download/{file_name}")
async def download_file(request: Request, file_name: str = F_Path(..., description="文件名称（含后缀）")):
    """分片下载文件，支持断点续传"""
    # 检查文件是否存在
    print(sys._getframe().f_lineno, request, file_name)
    file_path = Path(upload_file_path, file_name)
    #s1 = Path(file_path, file_name + '_0')
    #file_path = s1
    print(sys._getframe().f_lineno, file_path)
    if not os.path.exists(file_path):
        return {
            'code': 0,
            'error': '文件不存在'
            }

    # 获取文件的信息
    stat_result = os.stat(file_path)
    content_type, encoding = guess_type(file_path)
    content_type = content_type or 'application/octet-stream'
    print(sys._getframe().f_lineno, stat_result, content_type, encoding)

    # 读取文件的起始位置和终止位置
    range_str = request.headers.get('range', '')
    range_match = re.search(r'bytes=(\d+)-(\d+)', range_str, re.S) or re.search(r'bytes=(\d+)-', range_str, re.S)
    print(sys._getframe().f_lineno, range_str, range_match)

    if range_match:
        start_bytes = int(range_match.group(1))
        end_bytes = int(range_match.group(2)) if range_match.lastindex == 2 else stat_result.st_size - 1
    else:
        start_bytes = 0
        end_bytes = stat_result.st_size - 1
    # 这里 content_length 表示剩余待传输的文件字节长度
    # 这里 content_length 表示剩余待传输的文件字节长度
    content_length = stat_result.st_size - start_bytes if stat.S_ISREG(stat_result.st_mode) else stat_result.st_size
    print(sys._getframe().f_lineno, start_bytes, end_bytes, content_length)

    # 构建文件名称
    name, *suffix = file_name.rsplit('.', 1)
    suffix = f'.{suffix[0]}' if suffix else ''
    filename = quote(f'{name}{suffix}')  # 文件名编码，防止中文名报错
    print(sys._getframe().f_lineno, name, suffix, filename)
    # 打开文件从起始位置开始分片读取文件
    return StreamingResponse(
        file_iterator(file_path, start_bytes, 1024 * 1024 * 1),  # 每次读取 1M
        media_type=content_type,
        headers={
            'content-disposition': f'attachment; filename="{filename}"',
            'accept-ranges': 'bytes',
            'connection': 'keep-alive',
            'content-length': str(content_length),
            'content-range': f'bytes {start_bytes}-{end_bytes}/{stat_result.st_size}',
            'last-modified': formatdate(stat_result.st_mtime, usegmt=True),
        },
        status_code=206 if start_bytes > 0 else 200
    )
'''

def file_iterator(file_path, offset, chunk_size):
    """
    文件生成器
    :param file_path: 文件绝对路径
    :param offset: 文件读取的起始位置
    :param chunk_size: 文件读取的块大小
    :return: yield
    """
    print(sys._getframe().f_lineno)
    with open(file_path, 'rb') as f:
        f.seek(offset, os.SEEK_SET)
        while True:
            data = f.read(chunk_size)
            if data:
                yield data
            else:
                break


# model process
def run_model(path, output_dir):
    
    video_path = path
    print(sys._getframe().f_lineno, video_path)
    video = VideoProcessing(video_path)
    resultPrinter = ResultPrinter()
    if video.audio_time <= 4:
        raise ValueError('Audio time less than 4s, cannot infer its STI!')

    start_time = time.time()
    #ort_session = ort.InferenceSession(onnx_path)
    
    dataset = DatasetSlice(video.audio, video.fs, overlap, False)
    videoloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=vad_collate_fn, drop_last=False)
    print('Analyse STI of the audio...')
    output_collection = []
    empty_index = []
    with torch.no_grad():
        progress_bar = tqdm(videoloader)
        for i, datas in enumerate(progress_bar):
            images, temp_vad_lst = datas
            if temp_vad_lst:
                empty_index.extend(temp_vad_lst)
            if isinstance(images, bool):
                continue

            if not images.shape[0] == 4:
                continue
            input_image = images.numpy().astype(np.float32)
            # print("input image shape:",input_image.shape)

            output_pts = ort_session.run(None, {'modelInput': input_image})
            output_pts = np.squeeze(output_pts).tolist()
            # print("output pts shape:",output_pts)
            # onnx_cost  = time.time() - begin
            # print("pytorch output:",output_pts,",cost time:",pytorch_cost)
            # print("onnx output:",output_pts2,",onnx time:",onnx_cost)

            if isinstance(output_pts, float):
                output_collection.append(output_pts)
            else:
                output_collection.extend(output_pts)

    if empty_index:
        for item in empty_index:
            output_collection.insert(item, 'NaN')
        # print('STI Results:')
    resultPrinter.print_results(out=output_collection)
    save_name = os.path.join(output_dir, video_path.split('/')[-1].split('.')[0])
    # print("123132132132"+ video_path.split('\\')[-1].split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print("output_dir", output_dir)
    resultPrinter.save_csv_result(save_name + '.csv')
    print('- Csv results saved:', save_name + '.csv')

    gen_plot(resultPrinter.out_dict['STI'], resultPrinter.end_time_num, video_path, save_name + '.png')
    print('- Plot results saved:', save_name + '.png')

    count = 0
    while True:
        count += 1
        if count == 3:
            write(save_name + '.wav', video.fs, video.audio)
            print('- Automatically saved wav:', save_name + '.wav')
            break
        ifSaveAduio = 'y'#input('Do you want to save the audio file?(y/n)')
        if ifSaveAduio in ['y', 'Y']:
            write(save_name + '.wav', video.fs, video.audio)
            print('- Wav saved:', save_name + '.wav')
            break
        elif ifSaveAduio in ['n', 'N']:
            print('- You choose not to save the audio file.')
            break
        else:
            print('Please enter Y, y or N, n')


def gen_plot(result, endLst, file_path, save_path):
    if isinstance(result[0], float):
        filtered_res, timeLst = [result[0]], [0]
    else:
        filtered_res, timeLst = [], []

    sti_total = 0
    sti_count = 0
    for item, endtime in zip(result, endLst):
        if item not in ['nan', 'NaN']:
            sti_total += float(item)
            sti_count += 1
            filtered_res.append(float(item))
            timeLst.append(endtime)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    plt.xlabel('Time / s', fontsize=20)

    if sti_count != 0:
        avarage = str(sti_total / sti_count)[:4]
    else:
        avarage = 0

    plt.ylabel('STI', fontsize=20)

    plt.plot(timeLst, filtered_res, color='red', label='right', linestyle='-')

    plt.ylim(0, 1)
    plt.grid(linestyle='-.')
    plt.yticks(size=15)
    plt.xticks(size=15)

    yMajorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(yMajorLocator)

    xMajorLocator = MultipleLocator(10 ** int(np.log10(sti_count)))
    ax.xaxis.set_major_locator(xMajorLocator)

    title = "\n".join(['STI Result', 'File:' + file_path.split('/')[-1]])
    plt.suptitle(title, fontsize=25)
    fig.text(s='Average STI:  ' + str(avarage), x=0.85, y=0.955, fontsize=25, ha='center', color='red')
    fig.savefig(save_path)


class ResultPrinter(object):
    """一个无情的结果打印机器"""

    def __init__(self):
        self.out_dict = {'start': [], 'end': [], 'STI': []}
        self.end_time_num = []

    def print_results(self, out):
        already_time = self.get_start_len()
        for i in range(len(out)):
            start = (already_time + i) * overlap
            end = start + 4
            # print('time: [%.1f]s ~ [%.1f]s      STI: [%.2f] ' % (start, end, float(out[i])))
            self.out_dict['start'].append(second_to_time(start))
            self.end_time_num.append(start)
            self.out_dict['end'].append(second_to_time(end))
            # if i < average_iterval:
            #     average_sti = get_sum(out[:i + 1]) / (i + 1)
            # else:
            #     average_sti = get_sum(out[i - average_iterval + 1:i + 1]) / average_iterval
            self.out_dict['STI'].append(float(out[i]))

    def save_csv_result(self, save_file_path):
        for i in range(len(self.out_dict['STI'])):
            self.out_dict['STI'][i] = str(self.out_dict['STI'][i])[:4]
        out_data = pd.DataFrame.from_dict(self.out_dict)
        out_data.to_csv(save_file_path)
        print('Save csv already! Path: ', save_file_path)

    def get_start_len(self):
        return len(self.out_dict['start'])

class VideoProcessing(object):
    def __init__(self, path):
        self.path = path
        self.audio = None
        self.audio_time = None
        self.fs = None
        self.get_wav_from_video()

    def get_wav_from_video(self):
        print(sys._getframe().f_lineno, self.path)
        my_audio_clip = AudioFileClip(self.path)
        audio_data = my_audio_clip.to_soundarray()
        framerate = my_audio_clip.fps
        if framerate != 16000:
            audio_data = scipy.signal.resample(audio_data, int(len(audio_data) / framerate * 16000))
        nframes, nchannels = audio_data.shape
        if nchannels == 2:
            audio_data = audio_data.T[0]
        if isinstance(audio_data[0], np.float):
            audio_data = np.array(audio_data * 32768.0, dtype=np.int16)
        elif isinstance(audio_data[0], np.int32):
            audio_data = (audio_data >> 16).astype(np.int16)
        audio_time = len(audio_data) / 16000
        self.audio, self.audio_time, self.fs = audio_data, audio_time, 16000


def second_to_time(a):
    if '.5' in str(a):
        ms = 5000
        a = int(a - 0.5)
    else:
        ms = 0000
        a = int(a)
    h = a // 3600
    m = a // 60 % 60
    s = a % 60
    return str("{:0>2}:{:0>2}:{:0>2}.{:0>4}".format(h, m, s, ms))



if __name__ == '__main__':
    #ort_session = ort.InferenceSession(onnx_path)
    uvicorn.run(app=app, host="0.0.0.0", port=8000)

