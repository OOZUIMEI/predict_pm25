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
function getData(city, date){
    $.get("https://www.timeanddate.com/scripts/cityajax.php", 
        {
            "n": city,
            "mode":"historic",
            "hd": date,
            "month": parseInt(date.slice(4, 6)),
            "year": date.slice(0, 4),
            "json":"1"
        }, 
        function(res){
            console.save(res, city + "_" + date + ".json")
        }
    )    
}
function getAllFrom(date, city){
    var now = new Date()
    now = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    var st = new Date(date + 0)
    while(st <= now){
        var t1 = getDate(st, "")
        getData(city, t1)
        st.setDate(st.getDate() + 1)
    }
}
var jq = document.createElement('script')
jq.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"
document.getElementsByTagName('head')[0].appendChild(jq)

var city = "korea/daegu"
var fr = new Date("2017-01-01 00:00:00")

getAllFrom(fr, city)
