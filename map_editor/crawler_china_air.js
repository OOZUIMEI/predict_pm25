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
// var factors = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
var factors = ["pm2_5"]
var fr = "2017-06-01"
var en = "2018-03-01"
// save console to file

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
// get facetor data from d1 to d2
function getDataWithFactor(d1, d2, factor){
    $.post("http://www.ipe.org.cn/data_ashx/GetAirData.ashx", 
        {
        "headers[Cookie]":"ajaxkey=B4C669BE51C1E9577541E52C077B84515F2C74CA53418C00",
        "cmd":"getAirHistory",
        "mid":33,
        "indexname": factor,
        "type":0,
        "starttime":d1,
        "endtime":d2,
        "iscity":1
        }, 
        function(res){
            if(res.data && res.ser){
                var data = processData(d1, res)
                if(!data)
                    console.log(d1, res)
                else
                    console.save(data, factor + "_" + d1 + "_" + d2.replace(/\//g, "-") + ".csv")
            }
        },
        "json"
    )
}
function processData(d1, res){
    var st = new Date(d1 + " 00:00:00")
    st.setDate(st.getDate() + 1)
    var date = res.ser
    var data = res.data
    var str = ""
    for(var x = 1; x < date.length; x++){
        var ds = date[x].replace("日").replace("时").split("/")
        var h = (x - 1) % 24
        h = getFormat10(h)
        var d = parseInt(ds[0])
        var date_str = ""
        var dt = new Date(st.getFullYear(), st.getMonth(), d)
        date_str = getDate(dt, "-") + " " + h + ":00:00"
        str += date_str + "," + data[x] + "\n"
    }
    return str
}
// get data from d1 to d2
function getData(){
    var d1 = new Date(fr + " 00:00:00")
    var de = new Date(en + " 00:00:00")
    while(d1 < de){
        var td = new Date(d1.getFullYear(), d1.getMonth(), 0)
        var t1 = getDate(td, "-")
        var te = new Date(d1.getFullYear(), d1.getMonth() + 1, 0)
        d1 = new Date(d1.getFullYear(), d1.getMonth() + 1, 1)
        var t2 = getDate(te, "/")
        for(x in factors){
            getDataWithFactor(t1, t2, factors[x])
        }
        
    }    
}
getData()