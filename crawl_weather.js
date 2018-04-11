(function(console){

    console.save = function(data, filename){

        if(!data) {
            console.error('Console.save: No data')
            return;
        }

        if(!filename) filename = 'console.json'

        if(typeof data === "object"){
            data = JSON.stringify(data, undefined, 4)
        }

        var blob = new Blob([data], {type: 'text/json'}),
            e    = document.createEvent('MouseEvents'),
            a    = document.createElement('a')

        a.download = filename
        a.href = window.URL.createObjectURL(blob)
        a.dataset.downloadurl =  ['text/json', a.download, a.href].join(':')
        e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null)
        a.dispatchEvent(e)
    }
})(console)
function getFormat10(m){
    return m >= 10 ? (m + "") : ("0" + m)
}
// get date format
function getDate(td, fm){
    var m = td.getMonth() + 1
    var d = td.getDate()
    var mt = getFormat10(m)
    var dt = getFormat10(d)
    var t1 = td.getFullYear() + fm + mt + fm + dt
    return t1
}
function getData(key, city, date, url){
    $.post(url, 
        {
            "__VIEWSTATE": key,
            "__VIEWSTATEGENERATOR": "F960AAB1",
            "ctl00$rblTemp":1,
            "ctl00$rblPrecip":1,
            "ctl00$rblWindSpeed":1,
            "ctl00$rblPressure":1,
            "ctl00$rblVis":2,
            "ctl00$rblheight":1,
            "ctl00$MainContentHolder$txtPastDate": date,
            "ctl00$MainContentHolder$butShowPastWeather":"Get Weather"
        }, 
        function(res){
            console.save(res, city + "_" + date + ".html")
        }
    )    
}
function getAllFrom(key, date, city){
    var keys = {
        "beijing": "https://www.worldweatheronline.com/beijing-weather-history/beijing/cn.aspx",
        "seoul": "https://www.worldweatheronline.com/seoul-weather-history/kr.aspx",
        "daegu": "https://www.worldweatheronline.com/daegu-weather-history/kr.aspx"
    }
    var url = keys[city]
    var now = new Date()
    now = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    var st = new Date(date + 0)
    while(st <= now){
        var t1 = getDate(st, "-")
        getData(key, city, t1, url)
        st.setDate(st.getDate() + 1)
    }
}
var city = "daegu"
var fr = new Date("2017-01-01 00:00:00")
var key = "hGdXAgSS0GNOi3jGzUs2fyu5PMphYN2m1QIvVx0SbDUg0T0wXs76TGENoQGmTnGc0GduQEVYYY6r7GBxGqpTb5yrLvfYx+LhdH4+LaRCU7uN7sGn"
getAllFrom(key, fr, city)