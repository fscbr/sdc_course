/*
 * main.cpp
 * Reads in data and runs 2D particle filter.
 *  Created on: Dec 13, 2016
 *      Author: Tiffany Huang
 */

#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;
using Eigen::Vector3d;

void check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " [stdx stdy]";

  bool has_valid_args = false;

  if (argc == 1) {
    has_valid_args = true;//no parameters handed over
  } else if (argc == 4) {
    has_valid_args = true;//three parameters handed over
  } else if (argc > 4) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  } else {
    cerr << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
	check_arguments(argc, argv);

	double std_x;
	double std_y;
	double std_theta;

	/*
	 * Sigmas - just an estimate, usually comes from uncertainty of sensor, but
	 * if you used fused data from multiple sensors, it's difficult to find
	 * these uncertainties directly.
	 */
	double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
	double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]
	if(argc == 4)
	{
		sigma_pos[0] = atof(argv[1]);
		sigma_pos[1] = atof(argv[2]);
		sigma_pos[2] = atof(argv[3]);
	}

    // parameters related to grading.
    int time_steps_before_lock_required = 10; // number of time steps before accuracy is checked by grader.
    double max_runtime = 45; // Max allowable runtime to pass [sec]
    double max_translation_error = 1; // Max allowable translation error to pass [m]
    double max_yaw_error = 0.05; // Max allowable yaw error [rad]

    // Start timer.
    int start = clock();

    //Set up parameters here
    double delta_t = 0.1; // Time elapsed between measurements [sec]
    double sensor_range = 1000; // Sensor range [m]

    // noise generation
    default_random_engine gen;
    normal_distribution<double> N_obs_x(0, sigma_landmark[0]);
    normal_distribution<double> N_obs_y(0, sigma_landmark[1]);
    double n_x, n_y, n_theta;
    // Read map data
    Map map;
    if (!read_map_data("data/map_data.txt", map))
    {
        cout << "Error: Could not open map file" << endl;
        return -1;
    }

    // Read position data
    vector<control_s> position_meas;
    if (!read_control_data("data/control_data.txt", position_meas))
    {
        cout << "Error: Could not open position/control measurement file" << endl;
        return -1;
    }

    // Read ground truth data
    vector<ground_truth> gt;
    if (!read_gt_data("data/gt_data.txt", gt))
    {
        cout << "Error: Could not open ground truth data file" << endl;
        return -1;
    }

    // Run particle filter!
    int num_time_steps = position_meas.size();
    ParticleFilter pf;
    double total_error[3] = {0, 0, 0};
    double cum_mean_error[3] = {0, 0, 0};

    ofstream out_file_("output1.txt", ofstream::out);

    for (int i = 0; i < num_time_steps; ++i)
    {
        cout << "Time step: " << i << endl;
        // Read in landmark observations for current time step.
        ostringstream file;
        file << "data/observation/observations_" << setfill('0') << setw(6) << i + 1 << ".txt";
        vector<LandmarkObs> observations;
        if (!read_landmark_data(file.str(), observations))
        {
            cout << "Error: Could not open observation file " << i + 1 << endl;
            return -1;
        }

        // Initialize particle filter if this is the first time step.
        if (!pf.initialized())
        {
            pf.init(gt[0].x, gt[0].y, gt[0].theta, sigma_pos);

            //i starts with 1, so we can take that as well to predict
            pf.prediction(delta_t, sigma_pos, position_meas[i - 1].velocity, position_meas[i - 1].yawrate);
        }
        else
        {
            // predict the vehicle's next state (noiseless).
            pf.prediction(delta_t, sigma_pos, position_meas[i - 1].velocity, position_meas[i - 1].yawrate);
        }

        vector<LandmarkObs> noisy_observations;
        LandmarkObs obs;
        for (unsigned int j = 0; j < observations.size(); ++j)
        {
            n_x = N_obs_x(gen);
            n_y = N_obs_y(gen);
            obs = observations[j];
            obs.x = obs.x + n_x;
            obs.y = obs.y + n_y;
            noisy_observations.push_back(obs);
        }

        // Update the weights
        pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);

        vector<Particle> particles = pf.particles_;
        int num_particles = particles.size();

        //find the best particle
        Particle best_particle;
        double highest_weight = 0.0;
      	int best_index = -1;
       	for (int j = 0; j < num_particles; j++)
       	{
       		if (particles[j].weight > highest_weight)
       		{
       			highest_weight = particles[j].weight;
       			best_index = j;
       		}
       	}
       	best_particle = particles[best_index];

       	//resample after taking the best particle
        pf.resample();

        //
        double est_x = best_particle.pos(0);
        double est_y = best_particle.pos(1);
        double est_theta = best_particle.pos(2);

        cout << "Times step:" << i << " best particle:" << best_index << " pos:" << best_particle.pos.transpose() << " x:" << gt[i].x << " y:" << gt[i].y << endl;

        // Calculate and output the average weighted error of the particle filter over all time steps so far.
        double* avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, est_x , est_y, est_theta);
        for (int j = 0; j < 3; j++)
        {
            total_error[j] += avg_error[j];
            cum_mean_error[j] = total_error[j] / (double)(i + 1);
        }

        //logging for analysis
        ostringstream out_line;
        out_line << est_x << "\t" << est_y << "\t" << est_theta << "\t" << position_meas[i - 1].velocity << "\t" << position_meas[i - 1].yawrate << "\t";

        // output ground truth
    	out_line << gt[i].x << "\t" << gt[i].y << "\n";
        out_line.flush();
        out_file_ << out_line.str();

        // Print the cumulative weighted error
        cout << "Times step:" << i << " cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;

        // If the error is too high, say so and then exit.
        if (i >= time_steps_before_lock_required)
        {
            if (cum_mean_error[0] > max_translation_error || cum_mean_error[1] > max_translation_error || cum_mean_error[2] > max_yaw_error)
            {
                cout << "Times step: " << i << " " << sigma_pos[0] << " " << sigma_pos[1]<< " " << sigma_pos[2] << " cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2];
                if (cum_mean_error[0] > max_translation_error)
                {
                    cout << " Your x error, " << cum_mean_error[0] << " is larger than the maximum allowable error, " << max_translation_error << endl;
                }
                else if (cum_mean_error[1] > max_translation_error)
                {
                    cout << " Your y error, " << cum_mean_error[1] << " is larger than the maximum allowable error, " << max_translation_error << endl;
                }
                else
                {
                    cout << " Your yaw error, " << cum_mean_error[2] << " is larger than the maximum allowable error, " << max_yaw_error << endl;
                }
                return -1;
            }
        }
    }
    // close files
    if (out_file_.is_open()) {
      out_file_.close();
    }

    // Output the runtime for the filter.
    int stop = clock();
    double runtime = (stop - start) / double(CLOCKS_PER_SEC);
    cout << "Runtime (sec): " << runtime << endl;

    // Print success if accuracy and runtime are sufficient (and this isn't just the starter code).
    if (runtime < max_runtime && pf.initialized())
    {
        cout << "Success! Your particle filter passed!" << endl;
    }
    else if (!pf.initialized())
    {
        cout << "This is the starter code. You haven't initialized your filter." << endl;
    }
    else
    {
        cout << "Your runtime " << runtime << " is larger than the maximum allowable runtime, " << max_runtime << endl;
        return -1;
    }

    return 0;
}


