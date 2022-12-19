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

app = FastAPI(docs_url="/docs")

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_file_path = Path(base_dir, './uploads')


@app.post("/file-slice")
async def upload_file(
    request: Request,
    identifier: str = Body(..., description="文件唯一标识符"),
    number: str = Body(..., description="文件分片序号（初值为0）"),
    file: UploadFile = File(..., description="文件")
):
    """文件分片上传"""
    path = Path(upload_file_path, identifier)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = Path(path, f'{identifier}_{number}')
    if not os.path.exists(file_name):
        async with aiofiles.open(file_name, 'wb') as f:
            await f.write(await file.read())
    return {
        'code': 1,
        'chunk': f'{identifier}_{number}'
    }


@app.put("/file-slice")
async def merge_file(
    request: Request,
    name: str = Body(..., description="文件名称（不含后缀）"),
    file_type: str = Body(..., description="文件类型/后缀"),
    identifier: str = Body(..., description="文件唯一标识符")
):
    """合并分片文件"""
    target_file_name = Path(upload_file_path, f'{name}.{file_type}')
    path = Path(upload_file_path, identifier)
    try:
        async with aiofiles.open(target_file_name, 'wb+') as target_file:  # 打开目标文件
            for i in range(len(os.listdir(path))):
                temp_file_name = Path(path, f'{identifier}_{i}')
                async with aiofiles.open(temp_file_name, 'rb') as temp_file:  # 按序打开每个分片
                    data = await temp_file.read()
                    await target_file.write(data)  # 分片内容写入目标文件
    except Exception as e:
        return {
            'code': 0,
            'error': f'合并失败：{e}'
        }
    shutil.rmtree(path)  # 删除临时目录
    return {
        'code': 1,
        'name': f'{name}.{file_type}'
    }


@app.get("/file-slice/{file_name}")
async def download_file(request: Request, file_name: str = F_Path(..., description="文件名称（含后缀）")):
    """分片下载文件，支持断点续传"""
    # 检查文件是否存在
    print(sys._getframe().f_lineno, request, file_name)
    file_path = Path(upload_file_path, file_name)
    s1 = Path(file_path, file_name + '_0')
    file_path = s1
    print(sys._getframe().f_lineno, file_path, s1)
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


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
