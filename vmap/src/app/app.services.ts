import { Injectable } from "@angular/core"
import { HttpClient } from '@angular/common/http'

@Injectable({ providedIn: 'root' })
export class Services{

    private url = "http://localhost:8080/"

    constructor(private http: HttpClient){

    }

    getGeoProvince(){
        return this.http.get("assets/seoul_municipalities_geo.json")
    }

    getDistricts(){
        return this.http.get("assets/districts.json")
    }

    getPrediction(){
        return this.http.get(this.url + "prediction")
    }
}