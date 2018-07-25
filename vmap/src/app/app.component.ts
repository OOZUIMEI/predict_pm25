import { Component, ViewChild, OnInit } from '@angular/core';
import {MapComponent} from '@yaga/leaflet-ng2'
import { cloneDeep } from "lodash"
import { Services } from './app.services'
import {Chart} from 'angular-highcharts'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
//https://github.com/southkorea/southkorea-maps
//https://leafletjs.com/examples/choropleth/
//https://www.npmjs.com/package/angular-highcharts
export class AppComponent implements OnInit{

  private mapConfig: object = {
    minZoom: 10,
    maxZoom: 14,
    zoom: 12,
    lat: 37.5617,
    lng: 126.93923,
    tileLayerUrl: "http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    style: {
      weight: 1,
      opacity: 1,
      dashArray: '3',
      fillOpacity: 0.8
    }
  }

  // public iconAnchor: Point = new Point(30, 30)
  // public heat: Array<number> = [78,74,69,54,63,61,45,73,67,53,57,65,73,89,115,64,66,98,52,63,88,49,71,43,35]
  public district_geo: Object
  public prediction: Array<Object> = []
  public average: Array<number> = []
  public timestamp: Array<string> = []
  public current_time: number = 0
  public current_timestamp: string = ""
  public chart: Chart;
  public isMobile: boolean = false

  @ViewChild(MapComponent) private mapComponent: MapComponent;
  

  constructor(private services: Services){
    this.services.getGeoProvince().subscribe(
      res => {
        this.district_geo = res
      }
    )
    this.services.getPrediction().subscribe(
      res => {
        this.prediction = res["data"]
        this.average = res["avg"]
        this.timestamp = res["timestamp"]
        setTimeout(() => {
          this.select_prediction(this.current_time)
          for(let x in this.average){
            let d = Number(this.average[x])
            this.addPoint(d)
          }
        }, 500)
      }
    )
  }

  ngOnInit(){
    let sz = 150
    this.isMobile = this.checkMobile(navigator.userAgent)
    let wd = window.innerWidth
    if(wd <= 1199 && wd >= 768){
      sz = 140
      this.mapConfig.zoom = 11
      this.mapConfig.lng = 126.89923
    }else if(wd < 768){
      this.mapConfig.zoom = 10
      this.mapConfig.minZoom = 8
      this.mapConfig.lng = 126.96923
    }
    let chart = new Chart({
      chart: {
        type: 'column',
        height: sz,
      },
      title: {
        text: 'PM2.5 Inclination'
      },
      credits: {
        enabled: false
      },
      series: [{
        "name": 'Future Prediction(Hours)',
        "data": [],
        // keys: [1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
      }],
      tooltip: {
        headerFormat: '<span>+{point.key + 1} h</span><br/>',
        pointFormat: '<span>PM2.5</span>:<b>{point.y:.0f}</b>'
      },
      plotOptions: {
        column: {
          pointPadding: 0.001,
          groupPadding: 0.001,
          borderWidth: 0.5,
          // {"value":100,"color":"#ffff73"},{"value":150,"color":'#ff983d'},{"value":200,"color":'#ff312e'},{"value":300,"color":'#9000f0'},{"value":500,"color":'#851e35'},{"color":'#03c11d'}
          zones: [{"value": 0, "color": '#11f52c'},{"value": 5, "color": '#09f225'},{"value": 10, "color": '#08e824'},{"value": 15, "color": '#06df22'},{"value": 20, "color": '#05d521'},
                  {"value": 25, "color": '#04cb1f'},{"value": 30, "color": '#03c11d'},{"value": 35, "color": '#02b71c'},{"value": 40, "color": '#01ad1a'},{"value": 45, "color": '#01a319'},
                  {"value": 50, "color": '#ffff73'},{"value": 55, "color": '#fdfc67'},{"value": 60, "color": '#fbf95c'},{"value": 65, "color": '#f8f551'},{"value": 70, "color": '#f5f147'},
                  {"value": 75, "color": '#f2ec3d'},{"value": 80, "color": '#efe733'},{"value": 85, "color": '#ebe229'},{"value": 90, "color": '#e7dc20'},{"value": 95, "color": '#ded21c'},
                  {"value": 100, "color": '#ff983d'},{"value": 105, "color": '#fe9337'},{"value": 110, "color": '#fd8f31'},{"value": 115, "color": '#fc8a2c'},{"value": 120, "color": '#fb8526'},
                  {"value": 125, "color": '#fa8121'},{"value": 130, "color": '#f97c1b'},{"value": 135, "color": '#f87716'},{"value": 140, "color": '#f77210'},{"value": 145, "color": '#f66d0b'},
                  {"value": 150, "color": '#ff312e'},{"value": 155, "color": '#ff2926'},{"value": 160, "color": '#ff211f'},{"value": 165, "color": '#ff1917'},{"value": 170, "color": '#ff110f'},
                  {"value": 175, "color": '#ff0907'},{"value": 180, "color": '#ff0100'},{"value": 185, "color": '#f70100'},{"value": 190, "color": '#ef0100'},{"value": 195, "color": '#e80000'},
                  {"value": 200, "color": '#9000f0'},{"value": 205, "color": '#8e02eb'},{"value": 210, "color": '#8c03e6'},{"value": 215, "color": '#8905e1'},{"value": 220, "color": '#8707dc'},
                  {"value": 225, "color": '#8508d7'},{"value": 230, "color": '#830ad3'},{"value": 235, "color": '#810bce'},{"value": 240, "color": '#7f0dc9'},{"value": 245, "color": '#7c0ec5'},
                  {"value": 250, "color": '#7a0fc0'},{"value": 255, "color": '#7811bc'},{"value": 260, "color": '#7612b7'},{"value": 265, "color": '#7413b3'},{"value": 270, "color": '#7214ae'},
                  {"value": 275, "color": '#7015aa'},{"value": 280, "color": '#6d16a6'},{"value": 285, "color": '#6b17a1'},{"value": 290, "color": '#69189d'},{"value": 295, "color": '#671999'},
                  {"value": 300, "color": '#851e35'},{"value": 305, "color": '#851d34'},{"value": 310, "color": '#841c33'},{"value": 315, "color": '#841b32'},{"value": 320, "color": '#831a32'},
                  {"value": 325, "color": '#831931'},{"value": 330, "color": '#821830'},{"value": 335, "color": '#82172f'},{"value": 340, "color": '#81162e'},{"value": 345, "color": '#81152d'},
                  {"value": 350, "color": '#80142d'},{"value": 355, "color": '#7f142c'},{"value": 360, "color": '#7f132b'},{"value": 365, "color": '#7e122a'},{"value": 370, "color": '#7e112a'},
                  {"value": 375, "color": '#7d1029'},{"value": 380, "color": '#7c0f28'},{"value": 385, "color": '#7c0f27'},{"value": 390, "color": '#7b0e26'},{"value": 395, "color": '#7a0d26'},
                  {"value": 400, "color": '#7a0c25'},{"value": 405, "color": '#790c24'},{"value": 410, "color": '#780b24'},{"value": 415, "color": '#780a23'},{"value": 420, "color": '#770922'},
                  {"value": 425, "color": '#760922'},{"value": 430, "color": '#750821'},{"value": 435, "color": '#740720'},{"value": 440, "color": '#74071f'},{"value": 445, "color": '#73061f'},
                  {"value": 450, "color": '#72051e'},{"value": 455, "color": '#71051e'},{"value": 460, "color": '#70041d'},{"value": 465, "color": '#6f041c'},{"value": 470, "color": '#6f031c'},
                  {"value": 475, "color": '#6e031b'},{"value": 480, "color": '#6d021a'},{"value": 485, "color": '#6c021a'},{"value": 490, "color": '#6b0119'},{"value": 495, "color": '#6a0019'},
                  {"value": 500, "color": '#690018'}]
        }
      },
    });
    this.chart = chart
  }

  checkMobile(agent: string){
    if(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|mobile|CriOS/i.test(agent)){
      return true
    }
    return false
  }

  addPoint(point: number) {
    if (this.chart) {
      this.chart.addPoint(point);
    }
  }
  
  next(){
    if(this.current_time < (this.prediction.length - 1)){
      this.current_time++
      this.select_prediction(this.current_time)
    }
  }
  back(){
    if(this.current_time > 0){
      this.current_time --
      this.select_prediction(this.current_time)
    }
  }
  select_prediction(t: number){
    if(this.prediction.length > t){
      var i = 0;
      let data = this.prediction[t]
      let obj = cloneDeep(this.district_geo)
      for(var x in obj["features"]){
        var dis = obj["features"][x]
        dis["properties"]["density"] = Math.round(data[i])
        i++
      }
      this.mapComponent.invalidateSize()
      this.district_geo = obj
      this.current_timestamp = this.timestamp[t]
    }
  }

  style(geo: any, defaultStyle: any) {
    let getColor = function(d){
      let colors: Array<string> = ['#ffffff', '#11f52c', '#09f225', '#08e824', '#06df22', '#05d521', '#04cb1f', '#03c11d', '#02b71c', '#01ad1a', '#01a319', '#ffff73', '#fdfc67', '#fbf95c', '#f8f551', '#f5f147', '#f2ec3d', '#efe733', '#ebe229', '#e7dc20', '#ded21c', '#ff983d', '#fe9337', '#fd8f31', '#fc8a2c', '#fb8526', '#fa8121', '#f97c1b', '#f87716', '#f77210', '#f66d0b', '#ff312e', '#ff2926', '#ff211f', '#ff1917', '#ff110f', '#ff0907', '#ff0100', '#f70100', '#ef0100', '#e80000', '#9000f0', '#8e02eb', '#8c03e6', '#8905e1', '#8707dc', '#8508d7', '#830ad3', '#810bce', '#7f0dc9', '#7c0ec5', '#7a0fc0', '#7811bc', '#7612b7', '#7413b3', '#7214ae', '#7015aa', '#6d16a6', '#6b17a1', '#69189d', '#671999', '#851e35', '#851d34', '#841c33', '#841b32', '#831a32', '#831931', '#821830', '#82172f', '#81162e', '#81152d', '#80142d', '#7f142c', '#7f132b', '#7e122a', '#7e112a', '#7d1029', '#7c0f28', '#7c0f27', '#7b0e26', '#7a0d26', '#7a0c25', '#790c24', '#780b24', '#780a23', '#770922', '#760922', '#750821', '#740720', '#74071f', '#73061f', '#72051e', '#71051e', '#70041d', '#6f041c', '#6f031c', '#6e031b', '#6d021a', '#6c021a', '#6b0119', '#6a0019', '#690018']
      let color_range: Array<number> = [0, 0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500]
      if(d){
        let over_color = '#690018'
        for(var i = 0; i < color_range.length; i++){
          if(d <= color_range[i]){
            // stop here
            return colors[i - 1]
          }
        }
        return over_color
      }
      return "#ffffff"
    }
    defaultStyle["fillColor"] = getColor(geo['properties']['density'])
    return defaultStyle
  }

}
