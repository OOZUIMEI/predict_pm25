import {MapComponent, OSM_TILE_LAYER_URL} from '@yaga/leaflet-ng2'
import { Component, ViewChild, OnInit } from '@angular/core';
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
    zoom: 11,
    lat: 37.5917,
    lng: 126.99923,
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

  @ViewChild(MapComponent) private mapComponent: MapComponent;

  constructor(private services: Services){
    this.services.getGeoProvince().subscribe(
      res => {
        this.district_geo = res
        console.log(res)
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
            this.addPoint(Number(this.average[x]))
          }
        }, 500)
      }
    )
  }

  ngOnInit(){
    let chart = new Chart({
      chart: {
        type: 'line',
        height: 300,
      },
      title: {
        text: 'PM2.5 Inclination'
      },
      credits: {
        enabled: false
      },
      series: [{
        name: 'PM2.5',
        data: []
      }]
    });
    this.chart = chart
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
        dis["properties"]["density"] = Math.round(data[i]) + 15
        i++
      }
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
    defaultStyle["fillColor"] =  getColor(geo['properties']['density'])
    return defaultStyle
  }

}
