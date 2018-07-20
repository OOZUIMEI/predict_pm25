import {MapComponent, OSM_TILE_LAYER_URL} from '@yaga/leaflet-ng2'
import { Component, ViewChild } from '@angular/core';
import { cloneDeep } from "lodash"
import { Services } from './app.services'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
//https://github.com/southkorea/southkorea-maps
//https://leafletjs.com/examples/choropleth/
export class AppComponent{

  private mapConfig: object = {
    zoom: 11,
    lat: 37.5917,
    lng: 126.99923,
    tileLayerUrl: "http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    style: {
      weight: 1,
      opacity: 1,
      dashArray: '3',
      fillOpacity: 0.5
    }
  }

  // public iconAnchor: Point = new Point(30, 30)
  // public heat: Array<number> = [78,74,69,54,63,61,45,73,67,53,57,65,73,89,115,64,66,98,52,63,88,49,71,43,35]
  public district_geo: Object
  public prediction: Array<Object> = []
  public average: Array<number> = []
  public current_time: number = 0

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
        setTimeout(() => {
          this.select_prediction(this.current_time)
        }, 500)
      }
    )
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
      this.district_geo = obj
    }
  }

  style(geo: any, defaultStyle: any) {
    let getColor = function(d){
      return d > 400  ? '#690117' :
            d > 300  ? '#7a0f28' :
            d > 200  ? '#6f2aa1' :
            d > 150  ? '#d42926' :
            d > 100   ? '#fc8844' :
            d > 50   ? '#f0f04a' :
                        '#24d42a';
    }
    defaultStyle["fillColor"] =  getColor(geo['properties']['density'])
    return defaultStyle
  }

}
