# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from io import BytesIO
import os
import shutil
import subprocess
import tarfile
import traceback
from .lance import LANGUAGE_MODEL_HOME

if LANGUAGE_MODEL_HOME is None:
    raise Exception("LANCE_LANGUAGE_MODEL_HOME is not configured")

def check_lindera():
    if not shutil.which("lindera"):
        raise Exception("lindera is not installed. Please install it by following https://github.com/lindera/lindera/tree/main/lindera-cli")

def check_requests():
    try:
        import requests
    except:
        raise Exception("requests is not installed, Please pip install requests")

def download_jieba():
    dirname = os.path.join(LANGUAGE_MODEL_HOME, "jieba", "default")
    os.makedirs(dirname, exist_ok=True)
    try:
        check_requests()
        import requests
        resp = requests.get("https://api.github.com/repos/messense/jieba-rs/releases/latest")
        content = requests.get(resp.json()["tarball_url"]).content
        with tarfile.open(fileobj=BytesIO(content)) as tar:
            dir = tar.getnames()[0]
            tar.extract(f'{dir}/src/data', path=dirname)
        shutil.move(os.path.join(dirname, dir, "src", "data"), dirname)
    except Exception as _:
        traceback.print_exc()
        print("Download jieba language model failed. Please download this folder "
              f"https://github.com/messense/jieba-rs/tree/main/src/data and put it in {dirname}")

def download_lindera(lm: str):
    import requests
    dirname = os.path.join(LANGUAGE_MODEL_HOME, "lindera", lm)
    src_dirname = os.path.join(dirname, "src")
    if lm == "ipadic":
        url = "https://dlwqk3ibdg1xh.cloudfront.net/mecab-ipadic-2.7.0-20070801.tar.gz"
    elif lm == "ko-dic":
        url = "https://dlwqk3ibdg1xh.cloudfront.net/mecab-ko-dic-2.1.1-20180720.tar.gz"
    elif lm == "unidic":
        url = "https://dlwqk3ibdg1xh.cloudfront.net/unidic-mecab-2.1.2.tar.gz"
    else:
        raise Exception(f"language model {lm} is not supported")
    os.makedirs(src_dirname, exist_ok=True)
    print(f"downloading language model: {url}")
    data = requests.get(url).content
    print(f"unzip language model: {url}")

    cwd = os.getcwd()
    try:
        os.chdir(src_dirname)
        with tarfile.open(fileobj=BytesIO(data)) as tar:
            tar.extractall()
            name = tar.getnames()[0]
        cmd = ["lindera", "build", "--dictionary-kind=ipadic", os.path.join(src_dirname, name), dirname]
        print(f"compile language model: {' '.join(cmd)}")
        subprocess.run(cmd)
    finally:
        os.chdir(cwd)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Lance tokenizer language model downloader'
    )
    parser.add_argument('tokenizer', choices=['jieba', 'lindera'])
    parser.add_argument("-l", "--languagemodel")
    args = parser.parse_args()
    print(f"LANCE_LANGUAGE_MODEL_HOME={LANGUAGE_MODEL_HOME}")
    if args.tokenizer == 'jieba':
        download_jieba()
    elif args.tokenizer == 'lindera':
        download_lindera(args.languagemodel)

if __name__ == '__main__':
    main()

