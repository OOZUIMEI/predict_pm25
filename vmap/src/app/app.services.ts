import { Injectable } from "@angular/core"
import { HttpClient } from '@angular/common/http'

@Injectable({ providedIn: 'root' })
export class Services{

    private url = "http://127.0.0.1:8888/"

    constructor(private http: HttpClient){

    }

    getGeoProvince(){
        return this.http.get("assets/data/seoul_municipalities_geo.json")
    }

    getDistricts(){
        return this.http.get("assets/districts.json")
    }

    getPrediction(){
        return this.http.get(this.url + "prediction")
    }
}