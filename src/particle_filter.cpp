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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Number of particles
	num_particles=1000;

	// normal distribution using GPS measurements
	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	// sampling from the above distribution for 1000 particles
	for(int i=0;i<1000;i++){
		Particle p;
		p.id=i;
		p.weight=1;
		p.x=dist_x(gen);
		p.y=dist_y(gen);
		p.theta=dist_theta(gen);

		particles.push_back(p);
	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for(int i=0;i<num_particles;i++){
		//current values
		double x_current=particles[i].x;
		double y_current=particles[i].y;
		double theta_current=particles[i].theta;

		double x_pred,y_pred,theta_pred;
		// check if the motion is in straight line or not
		if(abs(yaw_rate)>1e-5){
			theta_pred=theta_current+yaw_rate*delta_t;
			x_pred=x_current+ velocity/yaw_rate * (sin(theta_pred)-sin(theta_current));
			y_pred=y_current+ velocity/yaw_rate * (cos(theta_current)-cos(theta_pred));

		}
		else{
			x_pred=x_current+velocity*cos(theta_current)*delta_t;
			y_pred=y_current+velocity*sin(theta_current)*delta_t;
			theta_pred=theta_current;
		}

		// normal distribution using motion model noises
		normal_distribution<double> dist_x(x_pred,std_pos[0]);
		normal_distribution<double> dist_y(y_pred,std_pos[1]);
		normal_distribution<double> dist_theta(theta_pred,std_pos[2]);

		// transition state
		particles[i].x=dist_x(gen);
		particles[i].y=dist_y(gen);
		particles[i].theta=dist_theta(gen);
	}



}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto& obs : observations) {
		double min_dist = numeric_limits<double>::max();

		for (const auto& pred_obs : predicted) {
			double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
			if (d < min_dist) {
				obs.id	 = pred_obs.id;
				min_dist = d;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double std_landmark_x=std_landmark[0];
	double std_landmark_y=std_landmark[1];

	for(int i=0;i<num_particles;i++){

		// get current state
		double p_x=particles[i].x;
		double p_y=particles[i].y;
		double p_theta=particles[i].theta;

		// get all the predicted landmarks in sensor range
		vector<LandmarkObs> landmarks_pred;
		for(const auto& landmark:map_landmarks.landmark_list){
			int landmark_id=landmark.id_i;
			double landmark_x=landmark.x_f;
			double landmark_y=landmark.y_f;

			double dist_to_landmark=dist(p_x,p_y,landmark_x,landmark_y);

			//check if the distance is within the sensor range
			if(dist_to_landmark<sensor_range){
				LandmarkObs land_pred;
				land_pred.id=landmark_id;
				land_pred.x=landmark_x;
				land_pred.y=landmark_y;

				// add to the vector of predicted landmarks
				landmarks_pred.push_back(land_pred);
			}

		}

		// transform the observation to map coordinate system
		vector<LandmarkObs> observed_landmarks_map;
		for(int i=0;i<observations.size();i++){
			LandmarkObs landmark_map;
			landmark_map.x=observations[i].x*cos(p_theta)-observations[i].y*sin(p_theta)+p_x;
			landmark_map.y=observations[i].x*sin(p_theta)+observations[i].y*cos(p_theta)+p_y;

			observed_landmarks_map.push_back(landmark_map);
		}

		// Data association problem
		dataAssociation(landmarks_pred,observed_landmarks_map);

		// Calculating the likelihood of every particle;
        double likelihood=1;

        double temp_mu_x,temp_mu_y;

        // iterate over all the observed land marks
        for (const auto& observation:observed_landmarks_map){
            // iterate over all the predicted landmarks
            for(const auto& landmark:landmarks_pred){
                // check for id matching
                if(observation.id==landmark.id){
                    temp_mu_x=landmark.x;
                    temp_mu_y=landmark.y;
                    break;
                }

            }
            double normalization_gaussian=2*M_PI*std_landmark_x*std_landmark_y;
            double unnormalizaed_prob=exp(-(pow(observation.x-temp_mu_x,2)/(2*std_landmark_x*std_landmark_y) +pow(observation.y-temp_mu_y,2)/(2*std_landmark_x*std_landmark_y) ));

            // joint likelihood  of all the landmarks is the production of the likelihood of every landmark as there are independant variables
            likelihood*=unnormalizaed_prob/normalization_gaussian;

        }

        // un normalized posterior distribution
        particles[i].weight=likelihood;

	}

    double weights_sum=0;
    for(const auto& item:particles){
        weights_sum+=item.weight;
    }

    // normalized weight (posterior distribution)

    for( auto& item:particles){
        item.weight/=(weights_sum +numeric_limits<double>::epsilon());
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<double> particle_weights;
	for (const auto& particle : particles)
		particle_weights.push_back(particle.weight);

	discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

	vector<Particle> resampled_particles;
	for (size_t i = 0; i < num_particles; ++i) {
		int k = weighted_distribution(gen);
		resampled_particles.push_back(particles[k]);
	}

	particles = resampled_particles;

	// Reset weights for all particles
	for (auto& particle : particles)
		particle.weight = 1.0;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
