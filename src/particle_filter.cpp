/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <math.h>

#define DEBUG_PREDICTED_LM 0
#define DEBUG_PREDICTED_OBS 0
#define DEBUG_PREDICTION 0
#include "particle_filter.h"
#define EPSILON 1e-4

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 50;
    default_random_engine gen;
    normal_distribution<double> xNoise(x,std[0]);
    normal_distribution<double> yNoise(y,std[1]);
    normal_distribution<double> yawNoise(theta,std[2]);

    for(int i=0; i < num_particles; ++i){
        Particle p = {i,xNoise(gen),yNoise(gen),yawNoise(gen), 1};
        particles.push_back(p);
        weights.push_back(1);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<double> xNoise(0,std_pos[0]);
    normal_distribution<double> yNoise(0,std_pos[1]);
    normal_distribution<double> yawNoise(0,std_pos[2]);

    for(int i = 0; i < num_particles; ++i){

        Particle p = particles[i];
        double yaw = p.theta;

        double delta_x = 0;
        double delta_y = 0; 
        double delta_yaw = 0;

        if(fabs(yaw_rate) < EPSILON){
            delta_x = velocity * delta_t * cos(yaw);
            delta_y = velocity * delta_t * sin(yaw);
        }
        else{
            double c = velocity/yaw_rate;
            delta_x = c * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
            delta_y = c * (cos(yaw) - cos(yaw + yaw_rate * delta_t) );
            delta_yaw = yaw_rate * delta_t;

        }
        //Add control noise
        delta_x += xNoise(gen);
        delta_y += yNoise(gen);
        delta_yaw += yawNoise(gen);
        //Add predcition
        
        if(DEBUG_PREDICTION){
            cout<<"Particle "<<i<<" before prediction ( "<< particles[i].x<<","<<particles[i].y<<","<<particles[i].theta<<")"<<endl;
            }

        particles[i].x += delta_x;
        particles[i].y += delta_y;
        particles[i].theta += delta_yaw;
        
        if(DEBUG_PREDICTION){
            cout<<"Particle "<<i<<" after prediction ( "<< particles[i].x<<","<<particles[i].y<<","<<particles[i].theta<<")"<<endl;
        }

    }
    

    
}
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for(int i = 0; i < observations.size(); ++i){
        int min = 1e6;
        int ld_id = -1;
        double x_obs = observations[i].x;
        double y_obs = observations[i].y;
        
        for(int j=0; j < predicted.size(); ++j){
            double x_pred = predicted[j].x;
            double y_pred = predicted[j].y;
            double distance = dist(x_pred,y_pred,x_obs,y_obs);
            if(distance < min){
                min = distance;
                ld_id = predicted[j].id;
            }
        }
        observations[i].id = ld_id;
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
    double weights_sum = 0;
    
    for(int j=0; j < num_particles; ++j){
        
        Particle p = particles[j];
        std::vector<LandmarkObs> predicted;
        std::map<int,int> lm2idx;
        std::vector<LandmarkObs> gObservations(observations.size());
        
        //Map observations from Particle frame to map frame
        for(int i = 0; i < observations.size(); ++i){
            double x = observations[i].x;
            double y = observations[i].y;
            gObservations[i].x = x * cos(p.theta) - y * sin(p.theta) + p.x ;
            gObservations[i].y = x * sin(p.theta) + y * cos(p.theta) + p.y ;
        }
        for(int k = 0; k < map_landmarks.landmark_list.size(); ++k){
            double x_lm = map_landmarks.landmark_list[k].x_f;
            double y_lm = map_landmarks.landmark_list[k].y_f;
            int id_lm = map_landmarks.landmark_list[k].id_i;
            double distance = dist(x_lm,y_lm,p.x,p.y);
            if(distance <= sensor_range){
                LandmarkObs obs = {id_lm , x_lm , y_lm};
                predicted.push_back(obs);
                lm2idx[id_lm] = k;
            }

        }
        if(DEBUG_PREDICTED_LM){
            for(int k = 0;k<predicted.size();++k){
                cout<<"Landmark "<<predicted[k].id<<" x:"<<predicted[k].x<<" y:"<<predicted[k].y<<endl;
            }
        }
     
        dataAssociation(predicted , gObservations);
        
        if(DEBUG_PREDICTED_OBS){
            for(int k = 0;k<gObservations.size();++k){
                cout<<"observation "<<k<<" associated with landmark "<< gObservations[k].id<<" x:"<<gObservations[k].x<<" y:"<<gObservations[k].y<<endl;
            }
        }

        double prob = 1.0;
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];
        double c = 1/(2*M_PI * std_x * std_y);
        
        for(int m=0; m < gObservations.size(); ++m){
            LandmarkObs obs = gObservations[m];
            Map::single_landmark_s lm;
            int landmark_id = obs.id; 
            lm = map_landmarks.landmark_list[lm2idx[landmark_id]];
            double x_obs = obs.x;
            double y_obs = obs.y;
            double x_lm = lm.x_f;
            double y_lm = lm.y_f;
           
            double x_diff = pow((x_obs - x_lm)/std_x,2.0);
            double y_diff = pow((y_obs - y_lm)/std_y,2.0);
            prob *= c*exp(-( x_diff + y_diff )/2);
            
        }

        weights[j] = prob;
        particles[j].weight = prob;
        weights_sum += prob;
    }

    //Normalize weights
    if(weights_sum > 0){
        for(int i = 0; i < weights.size(); ++i){
            weights[i] /= weights_sum;
//            if(weights[i] > 0)
//                cout<<i<<":"<<weights[i]<<endl;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles(particles.size());
    for(int i = 0; i < num_particles; ++i){
        int idx = d(gen);
        resampled_particles[i] = particles[idx];
    }
    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
