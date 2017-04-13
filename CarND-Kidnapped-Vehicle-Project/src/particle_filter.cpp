/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <math.h>
#include "particle_filter.h"
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	n_x_ = std::normal_distribution<double>(0, std[0]);
	n_y_ = std::normal_distribution<double>(0, std[1]);
	n_theta_ = std::normal_distribution<double>(0, std[2]);

	std::cout << "init pos x:" << x << " y:" << y << " theta:" << theta << std::endl;

    num_particles_ = 2;
	is_initialized_ = true;
	weights_ = std::vector<double>(num_particles_);
	for( int i = 0; i < num_particles_;i++)
	{
      weights_[i] = 1.0;
	  Particle particle;
	  particle.x = x +  n_x_(gen_);
	  particle.y = y + n_y_(gen_);
      particle.theta = theta +  n_theta_(gen_) * 2.0 * M_PI;

	  particle.weight = 1;

	  std::cout << "init particle " << i << " x:" << particle.x << " y:" << particle.y << " theta:" << particle.theta << endl;
	  particles_.push_back(particle);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	double yaw_rate_dt = yaw_rate*delta_t;
	double velocity_yaw_rate = velocity/yaw_rate;

    for(int i=0; i<num_particles_; i++) {
	    //Serial.println(particles_[i].theta);
	    particles_[i].x += velocity_yaw_rate * (sin(particles_[i].theta + yaw_rate_dt) - sin(particles_[i].theta));
	    particles_[i].y += velocity_yaw_rate * (cos(particles_[i].theta ) - cos(particles_[i].theta + yaw_rate_dt));
	    particles_[i].theta += yaw_rate_dt;

	    particles_[i].x += n_x_(gen_);
	    particles_[i].y += n_y_(gen_);
	    particles_[i].theta += n_theta_(gen_);

	    std::cout << "particle " << i
	    		<< " x:" << particles_[i].x
	    		<< " y:" << particles_[i].y
	    		<< " theta:" << particles_[i].theta << std::endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& observations, Map &map_landmarks) {
	// TODO: Find		Gaussian(mu, sigma, dist);
    // the observed measurement that is closest to each map landpark and assign the
	//   landmark to this particular observed landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

//	std::cout << "FIND NEAREST" << std::endl;
	for( int i =0; i < observations.size();i++)
	{

		double minDist = 99999.0;
		double nearest_o = -1;

		double x_p = observations[i].x;
		double y_p = observations[i].y;

		for( int j =0; j < map_landmarks.landmark_list.size();j++)
		{
			double x_o = map_landmarks.landmark_list[j].x_f;
			double y_o = map_landmarks.landmark_list[j].y_f;

			double dist_o_p = dist(x_p,y_p,x_o,y_o);

			if(dist_o_p < minDist)
			{
				minDist = dist_o_p;
				nearest_o = j;
			}

		}
		if(nearest_o > 0)
		{
// 		    std::cout << "ot x:" << observations[i].x << " ot y:" << observations[i].y;
			observations[i].x = map_landmarks.landmark_list[nearest_o].x_f;
			observations[i].y = map_landmarks.landmark_list[nearest_o].y_f;
//			std::cout << " m x:" << observations[i].x << " m y:" << observations[i].y << " dist:" << minDist << std::endl;
		}

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	//ist das rightig ich habe std in x und y -> brauche distance std?????
	double sigma = sqrt(std_landmark[0]*std_landmark[0] + std_landmark[1]*std_landmark[1]);
	double mu = 0;


	//find the real landmark values
	dataAssociation(observations, map_landmarks);

	for(int i = 0; i < num_particles_;i++)
	{
	  Particle p = particles_[i];

	  double prob = 1.0;


	  double x_p = p.x;
	  double y_p = p.y;

	  for(int j = 0; j < observations.size();j++)
	  {
		double x_o = observations[i].x;
		double y_o = observations[i].y;

		double g2D = gaussian2D(x_o,y_o,std_landmark[0],std_landmark[1], x_p,y_p);

//		std::cout << " g2D:" << g2D << std::endl;

		prob *= g2D;

	  }
	  std::cout << " particle:" << i << " weight:" << prob << std::endl;
	  p.weight = prob;
	  weights_[i] = prob;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	VectorXd weights = VectorXd(num_particles_, weights_);

	std::cout << " weights:" << weights << endl;;

	weights.normalize();
	weights *= 1000;

	std::cout << " weights:" << weights << endl;

	VectorXi weightsI = weights.cast<int>();

	std::discrete_distribution<> dw(weightsI.data(), weightsI.data()+weightsI.size());
	std::vector<Particle> new_particles;

    for(int i=0; i<num_particles_; i++) {
    	int pi = dw(gen_);
    	new_particles.push_back(particles_[pi]);
    	std::cout << "resample: " << pi << std::endl;
    }
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles_; ++i) {
		dataFile << particles_[i].x << " " << particles_[i].y << " " << particles_[i].theta << "\n";
	}
	dataFile.close();
}
