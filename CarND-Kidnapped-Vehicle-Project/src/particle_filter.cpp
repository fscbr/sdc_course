/*
 * particle_filter.cpp
 *
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

VectorXd ParticleFilter::positionNoise()
{
	VectorXd noise = VectorXd(3);
	noise << n_x_(gen_),n_y_(gen_),n_theta_(gen_);

	return noise;
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    std::cout << "init noise x:" << std[0] << " y:" << std[1] << " theta:" << std[2] << std::endl;

    n_x_ = std::normal_distribution<double>(0, std[0]);
    n_y_ = std::normal_distribution<double>(0, std[1]);
    n_theta_ = std::normal_distribution<double>(0, std[2]);

    std::cout << "init pos x:" << x << " y:" << y << " theta:" << theta << std::endl;

    num_particles_ = 100;
    is_initialized_ = true;
    weights_ = VectorXd(num_particles_);
    for ( int i = 0; i < num_particles_; i++)
    {
        weights_(i) = 1.0;
        Particle particle;
        particle.id = i;

        particle.pos = VectorXd(3);
        particle.pos <<  x , y, theta;
        particle.pos += positionNoise();
        particle.weight = 1;

        particles_.push_back(particle);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/


    for (int i = 0; i < num_particles_; i++)
    {
        particles_[i].pos = updatePosition(particles_[i].pos, velocity, yaw_rate, delta_t);
    	VectorXd noise = positionNoise();

    	particles_[i].pos += noise;

    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& observations, Map& map_landmarks)
{
    for (unsigned int i = 0; i < observations.size(); i++)
    {

        double minDist = 9999.0;
        unsigned int nearest_o = -1;

        double x_p = observations[i].x;
        double y_p = observations[i].y;

        //compare distances and mark the closest
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
        {
            double x_o = map_landmarks.landmark_list[j].x_f;
            double y_o = map_landmarks.landmark_list[j].y_f;

            double dist_o_p = dist(x_p, y_p, x_o, y_o);

            if (dist_o_p < minDist)
            {
                minDist = dist_o_p;
                nearest_o = j;
            }

        }
        //associate the best match
        if (nearest_o > 0)
        {
            observations[i].x = map_landmarks.landmark_list[nearest_o].x_f;
            observations[i].y = map_landmarks.landmark_list[nearest_o].y_f;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks)
{
    MatrixXd sigmaM = MatrixXd(2, 2);
    VectorXd obs = VectorXd(2);
    VectorXd xop = VectorXd(2);
    VectorXd mean = VectorXd(2);

    sigmaM << std_landmark[0], 0.0, 0.0, std_landmark[1];

    double min_Distance = 9999.0;
    int closest_particle = -1;

    double weight_sum = 0.0;
    for (int i = 0; i < num_particles_; i++)
    {
        double prob = 1.0;

        //deep copy observations to keep the original unchanged
        std::vector<LandmarkObs> conv_obs = observations;

        //convert observations relativ to particle position
        convertObservations(conv_obs, particles_[i].pos);

        //deep copy of converted observations
        std::vector<LandmarkObs> associations = conv_obs;

        //find the real landmark values
        dataAssociation(associations, map_landmarks);

        for (unsigned int j = 0; j < associations.size(); j++)
        {
            obs << conv_obs[j].x,conv_obs[j].y;
            mean << associations[j].x,associations[j].y;

            xop = obs - mean;

            //calc gaussian probability
            double gEigen = gaussianEigen( xop, sigmaM);

            prob *= gEigen;

        }

        //avoid too small numbers
        if (prob < 0.0000001)
        {
            prob = 0.0000001;
        }

        particles_[i].weight = prob;
        //sum of weights for normalization
        weight_sum += prob;
    }

    //normalize weights
    for (int i = 0; i < num_particles_; i++)
    {
        particles_[i].weight /= weight_sum;
        weights_(i) = particles_[i].weight;
    }
}

void ParticleFilter::resample()
{

	//discrete_distribution needs integer weights, so we scale them
    weights_ *= 100000;

    VectorXi weightsI = weights_.cast<int>();
//works only in linux
//    std::discrete_distribution<> dw(weightsI.data(), (weightsI.data() + weightsI.size()));
//works in visual studio and linux
    auto first = weightsI.data();
    auto last = weightsI.data() + weightsI.size();
    auto count = std::distance(first, last);
    std::discrete_distribution<> dw(
        count,
        -0.5,
        -0.5 + count,
        [&first](size_t i)
    {
        return *std::next(first, i);
    });

    //resample using the distribution
    std::vector<Particle> new_particles;
    for (int i = 0; i < num_particles_; i++)
    {
        int pi = dw(gen_);
        new_particles.push_back(particles_[pi]);
    }
    particles_ = new_particles;
}

void ParticleFilter::write(std::string filename)
{
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles_; ++i)
    {
        dataFile << particles_[i].pos.transpose() << "\n";
    }
    dataFile.close();
}
