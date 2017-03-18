# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)

[image1]: ./EKF-data1.png "result plot data1"
[image2]: ./EKF-data2.png "result plot data1"

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

## Measurement Covariance

For the Laser source measuring x,y position, I choosed:

| x       | y        |
|:-------:|:--------:|
| 0.00684 | 0        |
| 0       | 0.005489 |

For the Rader source measuring range, bearing, range rate, I choosed:

| range  | bearing  | rate  |
|:------:|:--------:|:-----:|
| 0.0144 | 0        | 0     |
| 0      | 0.000001 | 0     |
|0       |0         | 0.011 |

## Results

For the two data files I got these Accuracy - RMSE:

|data1|data2|
|:-----:|:-----:| 
|0.0337424|0.17585|
|0.0275774|0.167605|
|0.458483|0.384221|
|0.44524|0.538389|

Data 1 result image:

![alt text][image1]

Data 2 result image:

![alt text][image2]


