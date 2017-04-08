# Unscented Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)

[image1]: ./UKF-data1.png "result plot data1"
[image2]: ./UKF-data2.png "result plot data1"

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./Unscented ../data/sample-laser-radar-measurement-data-1.txt output.txt`

## Measurement Covariance

For the Laser source measuring x,y position, I choosed:

| x       | y        |
|:-------:|:--------:|
| 0.05    | 0        |
| 0       | 0.05     |

For the Rader source measuring range, bearing, range rate, I choosed:

| range  | bearing  | rate  |
|:------:|:--------:|:-----:|
| 0.0606 | 0        | 0     |
| 0      | 0.2535   | 0     |
| 0      |0         | 0.1757|

For the process noise standard deviation longitudinal acceleration I choosed 0.5306, 
for the process noise standard deviation yaw acceleration 0.6541

## Results

For the two data files I got these Accuracy - RMSE:

| data1 | data2 |
|:-----:|:-----:| 
| 0.0597718| 0.194016 |
| 0.0539679| 0.191292 |
| 0.544135 | 0.355867 |
| 0.526748 | 0.532885  |

Data 1 result image:

![alt text][image1]

Data 2 result image:

![alt text][image2]


