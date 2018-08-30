Regular model training
# train normal:

python train.py -pr vectors/full_256_sep/trainlb2.txt -f vectors/full_256_sep/train_256_p2 -fw "basic"  -l mae -r 10 -e 256 -bs 32 -sl 24 -s 20 -ds 249 -il 1 -ci 0 -rn 1 -p "32_256_24h_20h_full_pm25cn_tf_fromfull"

# train transfer:

python train.py -pr vectors/full_256_sep/trainlb2 -f vectors/full_256_sep/train_256_p2 -fw "basic" -pr1 weights/32_256_24h_20h_full_pm25cndaegu.weights -l mae -r 10 -e 256 -bs 32 -sl 24 -s 20 -ds 249 -il 1 -ci 0 -rn 1 -p "32_256_24h_20h_full_pm25cn_tf_fromfull"

# test:
python test.py -f vectors/full_256_sep/test_256_p2 -pr vectors/full_256_sep/testlb2.txt -wurl weights/32_256_24h_20h_full_pm25cndaegu.weights  -sl 24 -r 10 -e 256 -il 1 -ds 249 -bs 4 -ci 0 -rn 1 -p "32_test_24h_24h_fullcn" -s 24

# train seoul only
python train.py -pr vectors/seoul_data/train_labels.txt -f vectors/seoul_data/train -e 190 -bs 32 -sl 24 -s 20 -ds 184 -rn 2 -p "32_190_seoul_24_20"

Crawling:
# crawl autonomous weather system
    $ nohup python craw_aws.py -t 0 -s "2010-01-01 00:00:00" -e "2011-12-31 00:00:00" -i 1 -si 10 &
    # crawl continously
    $ python craw_aws.py -f 1 -s "2018-07-19 00:00:00" -i 1 -f 1

# crawl air pollution 
    $ nohup python craw_seoul_aqi.py -t 0 -s "2010-01-01 00:00:00" -e "2011-12-31 00:00:00" -i 1 -si 10 &
    # continously
    $ python craw_seoul_aqi.py -f 1 -s "2018-07-20 00:00:00" -i 1

crawl daegu_aqi

# crawl weather
    $ python craw_weather.py -f 1

CNN - LSTM Training


PROCESS SPATIOTEMPORAL
- Convert array to vectors bin
python process_sp_vector.py -u raw/sp_seoul_test -u1 sp_seoul_test

- Convert Data to Vectors of china:
python process_vectors.py -u ~/Documents/datasets/spatio_temporal_ck/raw/sample_china -u1 ~/Documents/datasets/spatio_temporal_ck/sample_china_bin -t 2 -dim 17

- Convert bin to grid SEOUL - CHINA
python process_sp_vector.py -u vectors/spatiotemporal/china_combined/sp_seoul_test_bin -au vectors/spatiotemporal/china_combined/sp_china_test_bin -u1 vectors/spatiotemporal/china_combined/test -au1 vectors/spatiotemporal/china_combined/test_china -t 1
python process_sp_vector.py -u ~/Documents/datasets/spatio_temporal_ck/sample_seoul_bin -u1 ~/Documents/datasets/spatio_temporal_ck/sample_seoul_grid -t 1


Train Spatiotemporal Seoul-China
python train_sp.py -u vectors/spatiotemporal/sample_sp_grid -au vectors/spatiotemporal/china_combined/sp_china_test_bin -w "test_gan" -f 1 -e 15 -ds 9

# start server

ng serve --port 3000 --host 0.0.0.0
