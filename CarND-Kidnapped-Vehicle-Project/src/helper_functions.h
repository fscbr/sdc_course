/*
 * helper_functions.h
 * Some helper functions for the 2D particle filter.
 *  Created on: Dec 13, 2016
 *      Author: Tiffany Huang
 */

#ifndef HELPER_FUNCTIONS_H_
#define HELPER_FUNCTIONS_H_

#define _USE_MATH_DEFINES
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "map.h"
#include <cstdlib>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Struct representing one position/control measurement.
 */
struct control_s
{

    double velocity;	// Velocity [m/s]
    double yawrate;		// Yaw rate [rad/s]
};

/*
 * Struct representing one ground truth position.
 */
struct ground_truth
{

    double x;		// Global vehicle x position [m]
    double y;		// Global vehicle y position
    double theta;	// Global vehicle yaw [rad]
};

/*
 * Struct representing one landmark observation measurement.
 */
struct LandmarkObs
{

    int id;				// Id of matching landmark in the map.
    double x;			// Local (vehicle coordinates) x position of landmark observation [m]
    double y;			// Local (vehicle coordinates) y position of landmark observation [m]
};

/*
 * Computes the Euclidean distance between two 2D points.
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output Euclidean distance between two 2D points
 */
inline double dist(double x1, double y1, double x2, double y2)
{
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

inline double* getError(double gt_x, double gt_y, double gt_theta, double pf_x, double pf_y, double pf_theta)
{
    static double error[3];
    error[0] = fabs(pf_x - gt_x);
    error[1] = fabs(pf_y - gt_y);
    error[2] = fabs(pf_theta - gt_theta);
    return error;
}

/* Reads map data from a file.
 * @param filename Name of file containing map data.
 * @output True if opening and reading file was successful
 */
inline bool read_map_data(std::string filename, Map& map)
{

    // Get file of map:
    std::ifstream in_file_map(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file.
    if (!in_file_map)
    {
        return false;
    }

    // Declare single line of map file:
    std::string line_map;

    // Run over each single line:
    while (getline(in_file_map, line_map))
    {

        std::istringstream iss_map(line_map);

        // Declare landmark values and ID:
        float landmark_x_f, landmark_y_f;
        int id_i;

        // Read data from current line to values::
        iss_map >> landmark_x_f;
        iss_map >> landmark_y_f;
        iss_map >> id_i;

        // Declare single_landmark:
        Map::single_landmark_s single_landmark_temp;

        // Set values
        single_landmark_temp.id_i = id_i;
        single_landmark_temp.x_f  = landmark_x_f;
        single_landmark_temp.y_f  = landmark_y_f;

        // Add to landmark list of map:
        map.landmark_list.push_back(single_landmark_temp);
    }
    return true;
}

/* Reads control data from a file.
 * @param filename Name of file containing control measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_control_data(std::string filename, std::vector<control_s>& position_meas)
{

    // Get file of position measurements:
    std::ifstream in_file_pos(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file.
    if (!in_file_pos)
    {
        return false;
    }

    // Declare single line of position measurement file:
    std::string line_pos;

    // Run over each single line:
    while (getline(in_file_pos, line_pos))
    {

        std::istringstream iss_pos(line_pos);

        // Declare position values:
        double velocity, yawrate;

        // Declare single control measurement:
        control_s meas;

        //read data from line to values:

        iss_pos >> velocity;
        iss_pos >> yawrate;


        // Set values
        meas.velocity = velocity;
        meas.yawrate = yawrate;

        // Add to list of control measurements:
        position_meas.push_back(meas);
    }
    return true;
}

/* Reads ground truth data from a file.
 * @param filename Name of file containing ground truth.
 * @output True if opening and reading file was successful
 */
inline bool read_gt_data(std::string filename, std::vector<ground_truth>& gt)
{

    // Get file of position measurements:
    std::ifstream in_file_pos(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file.
    if (!in_file_pos)
    {
        return false;
    }

    // Declare single line of position measurement file:
    std::string line_pos;

    // Run over each single line:
    while (getline(in_file_pos, line_pos))
    {

        std::istringstream iss_pos(line_pos);

        // Declare position values:
        double x, y, azimuth;

        // Declare single ground truth:
        ground_truth single_gt;

        //read data from line to values:
        iss_pos >> x;
        iss_pos >> y;
        iss_pos >> azimuth;

        // Set values
        single_gt.x = x;
        single_gt.y = y;
        single_gt.theta = azimuth;

        // Add to list of control measurements and ground truth:
        gt.push_back(single_gt);
    }
    return true;
}

/* Reads landmark observation data from a file.
 * @param filename Name of file containing landmark observation measurements.
 * @output True if opening and reading file was successful
 */
inline bool read_landmark_data(std::string filename, std::vector<LandmarkObs>& observations)
{

    // Get file of landmark measurements:
    std::ifstream in_file_obs(filename.c_str(), std::ifstream::in);
    // Return if we can't open the file.
    if (!in_file_obs)
    {
        return false;
    }

    // Declare single line of landmark measurement file:
    std::string line_obs;

    // Run over each single line:
    while (getline(in_file_obs, line_obs))
    {

        std::istringstream iss_obs(line_obs);

        // Declare position values:
        double local_x, local_y;

        //read data from line to values:
        iss_obs >> local_x;
        iss_obs >> local_y;

        // Declare single landmark measurement:
        LandmarkObs meas;

        // Set values
        meas.x = local_x;
        meas.y = local_y;

        // Add to list of control measurements:
        observations.push_back(meas);
    }
    return true;
}

inline double gaussianEigen(VectorXd x, MatrixXd sigma)
{
    return exp(-0.5 * x.transpose() * sigma.inverse() * x ) / (2 * M_PI * sqrt(sigma(0, 0) * sigma(1, 1)));
}

inline void convertObservations(std::vector<LandmarkObs>& observations, VectorXd position)
{

    double x = position(0);
    double y = position(1);
    double theta = position(2);
    //convert observatuions. hier brauche ich die vecicle position
    //  std::cout << "gt x:" << gt.x << " gt y:" << gt.y << " gt theta:" << gt.theta << std::endl;
    for (unsigned int i = 0; i < observations.size(); i++)
    {
        double conv_x = observations[i].x * cos(theta) - observations[i].y * sin(theta) + x;
        double conv_y = observations[i].x * sin(theta) + observations[i].y * cos(theta) + y;

//        std::cout << "x:" << observations[i].x << " y:" << observations[i].y;

        observations[i].x = conv_x;
        observations[i].y = conv_y;

//        std::cout << " cx:" << conv_x << " cy:" << conv_y << std::endl;

    }
}

inline VectorXd updatePosition(const VectorXd pos,const  double velocity,const  double yaw_rate,const double delta_t)
{
  double yaw_rate_dt = yaw_rate * delta_t;
  double velocity_yaw_rate = velocity / yaw_rate;
  double velocity_dt = velocity * delta_t;

  VectorXd newPos = VectorXd(3);

  double theta = pos(2);

  //avoid division by zero
  if (abs(yaw_rate) > 0.0001) {
    newPos(0)  = pos(0) + velocity_yaw_rate * (sin(theta + yaw_rate_dt) - sin(theta));
    newPos(1)  = pos(1) + velocity_yaw_rate * (cos(theta) - cos(theta + yaw_rate_dt));
  } else {
	  newPos(0) = pos(0) + velocity_dt * cos(theta);
	  newPos(1) = pos(1) + velocity_dt * sin(theta);
  }

  newPos(2)  = theta  +  yaw_rate_dt;
  return newPos;
}

inline void testGaussianEigen()
{

    MatrixXd sigma = MatrixXd(2, 2);
    sigma << 0.3, 0, 0, 0.3;

    std::cout << "i;j1;j2;j3;j4;j5;j6;j7;j8;j9;j10;j11" << std::endl;
    for (int j = 0; j < 11; j++)
    {
        std::cout << ";" << (j - 5) / 5.;
    }
    std::cout << std::endl;

    for (int i = 0; i < 11; i++)
    {
        std::cout << (i - 5) / 5. ;
        for (int j = 0; j < 11; j++)
        {
            VectorXd x = VectorXd(2);
            x << (i - 5.) / 5, (j - 5) / 5.;
            double p = gaussianEigen(x, sigma);
            std::cout << ";" << p;
        }
        std::cout << std::endl;
    }
}

inline double mapDistance(std::vector<LandmarkObs>& observations, std::vector<LandmarkObs>& associations)
{
	double dist_o_a = 0.0;
	for (unsigned int j = 0; j < observations.size(); j++)
    {
       double x_a = associations[j].x;
       double y_a = associations[j].y;
       double x_o = observations[j].x;
       double y_o = observations[j].y;

       dist_o_a += dist(x_a, y_a, x_o, y_o);
    }
	return dist_o_a;
}

inline VectorXd mapDistanceVector(std::vector<LandmarkObs>& associations, std::vector<LandmarkObs>& observations)
{
	VectorXd dist_o_a = VectorXd (2);
	dist_o_a << 0.0,0.0;
	for (unsigned int j = 0; j < observations.size(); j++)
    {
  	   VectorXd d = VectorXd (2);
       d << (associations[j].x - observations[j].x), (associations[j].y - observations[j].y);
       dist_o_a += d;
    }
	return dist_o_a;
}

#endif /* HELPER_FUNCTIONS_H_ */

