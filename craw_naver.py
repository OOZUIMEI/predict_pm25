import requests
import utils
"""

var x = $.ajax({
"url": "https://map.naver.com/api2.map/traffic/getEncryptedURLString.nhn",
"data": {
    "caller": "naver_map",
    "output": "json",
    "cctvId": "5018",
    "useEnableCctv": "true",
    "callback": "?",
    "type": "jsonp",
    "_": new Date().getTime()
}
})
var data = x.responseText
data = data.replace("jQuery181040398993664170213_1545286489231(", "").replace(");","")
data = JSON.parse(data)
var video_str = data.message.result.encryptedString
# 5018 is cctv id
var link_video = "https://ssl.map.naver.com/cctvsec.ktict/5018/" + video_str

"""
data = {
    "caller": "naver_map",
    "output": "json",
    "cctvId": "5018",
    "useEnableCctv": "true",
    "callback": "?",
    "type": "jsonp",
    "_": str(utils.now_milliseconds()) + ""
}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

r = requests.get("https://map.naver.com/api2.map/traffic/getEncryptedURLString.nhn?caller=naver_map&output=json&cctvId=5018&useEnableCctv=true&callback=?", headers=headers)
print(r.text)