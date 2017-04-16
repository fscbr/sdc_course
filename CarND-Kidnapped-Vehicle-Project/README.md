# Particle Filter Project
Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)

[image1]: ./partikelFilter-data.png "result plot data"

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make && cd ..`
4. Run it: `build/particle_filter` 

## Measurement Covariance

For the GPS measurement uncertainty, I choosed:

| x       | y        | theta |
|:-------:|:--------:|:-----:|
| 0.3     | 0        | 0     |
| 0       | 0.3      | 0     |
| 0       | 0        | 0.01  |

For the Landmark measurement uncertainty, I choosed:

| x      | y     |
|:------:|:-----:|
| 0.3    | 0     | 
| 0      | 0.3   | 

## Results

For the data file I got these cumulative errors: x 0.119899 y 0.112112 yaw 0.00387415

Data result image:

![alt text][image1]


