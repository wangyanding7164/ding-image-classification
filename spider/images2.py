import re
import os
import requests


# 获取网站源代码
def get_html(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    response.encoding = "utf-8"
    if response.status_code == 200:
        return response.text

    else:
        print("网站源码获取错误")


def parse_pic_url(html):
    result = re.findall('thumbURL":"(.*?)"', html, re.S)
    return result


def get_pic_content(url):
    response = requests.get(url)
    return response.content


def save_pic(fold_name,content, pic_name):
    with open(fold_name+"/" + str(pic_name) + ".jpg", "wb") as f:
        f.write(content)
        f.close()


# 定义一个文件夹
def create_fold(fold_name):
    try:
        os.mkdir(fold_name)
    except:
        print("文件已存在")


def main():
    try:
        # 输入文件夹名字
        fold_name = input("请输入您要抓取图片的名字")
        # 调用函数，创建文件夹
        create_fold(fold_name)
        # 输入要抓取的图片页数
        page_name = input("请输入要抓取的页数  （0，1，2，3，4......）")
        pic_name = 0
        for i in range(10):
            try:
                url = "https://image.baidu.com/search/acjson?tn=resultjson_com&logid=11836243366050550448&ipn=rj&ct=201326592&is=&fp=result&fr=ala&word=%E5%A4%A7%E7%86%8A%E7%8C%AB&queryWord=%E5%A4%A7%E7%86%8A%E7%8C%AB&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=&expermode=&nojc=&isAsync=&pn=60&rn=30&gsm=3c&1695869692997="
                headers = \
                    {"Accept": "text/plain, */*; q=0.01",
                     "Accept-Encoding": "gzip, deflate",
                     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                     "Connection": "keep-alive",
                     "Cookie": "winWH=%5E6_1659x838; BDIMGISLOGIN=0; BDqhfp=%E5%A4%A7%E7%86%8A%E7%8C%AB%26%26-10-1undefined%26%268568%26%267; BIDUPSID=84AA588D485BC5D9748C16152F786E4A; PSTM=1664863489; BDUSS=9UelhFRmVxQ2FYRURpM2hnanRSb09DcE5BcDFIYmdhM25DSXd3bWFMLX5mbWhqRVFBQUFBJCQAAAAAAAAAAAEAAABc%7EUGiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL%7ExQGO%7E8UBjc2; BDUSS_BFESS=9UelhFRmVxQ2FYRURpM2hnanRSb09DcE5BcDFIYmdhM25DSXd3bWFMLX5mbWhqRVFBQUFBJCQAAAAAAAAAAAEAAABc%7EUGiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL%7ExQGO%7E8UBjc2; BAIDUID=AA120298DBC668808E941F202EDAFE7D:FG=1; BAIDUID_BFESS=AA120298DBC668808E941F202EDAFE7D:FG=1; ZFY=ZkM1wYgsnkzHUCE:B8RSn0l9c2wZElo2ztkkXles7ZEQ:C; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; cleanHistoryStatus=0; BDRCVFR[Tp5-T0kH1pb]=mk3SLVN4HKm; BDRCVFR[tox4WRQ4-Km]=mk3SLVN4HKm; indexPageSugList=%5B%22%E5%A4%A7%E7%86%8A%E7%8C%AB%22%2C%22%E7%9C%BC%E9%95%9C%E6%A1%86%E7%A2%8E%22%5D; userFrom=null; ab_sr=1.0.1_ZjU4YWMxNDUwYzdmOTA5MzNlOTcwMzU1Y2Q2Yzg5N2EyNDAxYTJmY2E1NGU4MTFjZDYzMDllMmQ1ZTcyYzE2NmJhNTNmY2I3YzAyOWNkZDEzYzhiMmRlMWUxMWEzMTdiNGNkZTEzNTk3N2JiOGY2NjUxZTYyZGYwMTYwNTkzZWI3YWU1MmVmMThhNWU5ZWMwYThkYmIyY2UxNWFhM2RiZg==",
                     "Host": "image.baidu.com",
                     "Referer": "https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gb18030&word=%B4%F3%D0%DC%C3%A8&fr=ala&ala=1&alatpl=normal&pos=0&dyTabStr=MTEsMCwxLDYsMyw1LDQsMiw4LDcsOQ%3D%3D",
                     "Sec-Ch-Ua": '"Microsoft Edge";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
                     "Sec-Ch-Ua-Mobile": "?0",
                     "Sec-Ch-Ua-Platform": '"Windows"',
                     "Sec-Fetch-Dest": "empty",
                     "Sec-Fetch-Mode": "cors",
                     "Sec-Fetch-Site": "same-origin",
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.43",
                     "X-Requested-With": "XMLHttpRequest", }
                params = {"tn": "resultjson_com",
                          "logid": "11836243366050550448",
                          "ipn": "rj",
                          "ct": "201326592",
                          "fp": "result",
                          "fr": "ala",
                          "word": fold_name,
                          "queryWord": fold_name,
                          "cl": "2",
                          "lm": "-1",
                          "ie": "utf-8",
                          "oe": "utf-8",
                          "pn": str(int(i + 1) * 30),
                          "rn": "30",
                          "gsm": "3c"
                          }
                html = get_html(url, headers, params)
                result = parse_pic_url(html)
                for item in result:
                    pic_content = get_pic_content(item)
                    save_pic(fold_name,pic_content, pic_name)
                    pic_name += 1
                    print("正在保存第" + str(pic_name) + "张图片")
            except:
                print("抓取第"+str()+"页错误")
    except:
        print("图片抓取异常.....")


if __name__ == "__main__":
    main()
