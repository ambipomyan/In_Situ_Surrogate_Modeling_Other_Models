# Linear_Exponential_Regression
Algorithm for curve fitting of format: A\*exp(-alpha\*t) + B\*exp(-beta\*t) + C\*exp(-gamma\*t)

### TODO- use LinearRegression
```
python location_velocity_prediction_exp_params.py --src data/LULESH_Vel_L1_30_932.csv --loc 1 --min 0 --max 200 --alpha 0.16 --beta 0.05 --gamma 0.006
```
```
0         0.000000
1       967.470440
2      1188.487213
3      1344.647498
4      1462.654945
          ...     
927      10.970542
928      10.951030
929      10.930410
930      10.917282
931      10.906746
Name: l1, Length: 932, dtype: float64
0 200 932
0.16 0.05 0.006
[-3293.10069859  3397.95394452   345.52170367]
13.842953775361366
MSE-overall: 970.1760913968026
rate-overall: 0.10148288754864793
MSE-after-max: 22.537373046171812
rate-after-max: 0.0974929631551746
```

### Run Script
#### location velocity prediction
```
python location_velocity_prediction.py --src data/LULESH_Vel_L1_10.csv --loc 1 --p1 13 --p2 60 --min 0 --max 400
```
#### domain velocity prediction
```
python domain_velocity_prediction.py --src data/LULESH_Vel_L1_10.csv --min 0 --max 400 --start 4 --dist 4 --loc 5 --nlag 50
```

### Preliminary Results
https://docs.google.com/spreadsheets/d/1XyADeciUFuSKxHqoLBtG68VV8jSx_UykS21_Y3uyu9Y/edit#gid=546843254
