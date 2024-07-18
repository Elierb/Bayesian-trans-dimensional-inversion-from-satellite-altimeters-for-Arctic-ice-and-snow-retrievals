#include <vector>
#include <stdio.h>
#include <math.h>

#include "genericinterface.hpp"

typedef enum {
    SNOW_MODEL = 0,
    ICE_MODEL = 1,
    CS2_PENETRATION_MODEL = 2
    //AK_PENETRATION_MODEL = 2
}

sealand_model_t;

const float DENSITY_WATER = 1023.0;
//const float DENSITY_ICE = 917.0;
//const float DENSITY_SNOW = 300.0;
const float SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW = 1.28;


typedef enum {
    CS2_FREEBOARD_OBS = 0,
    AK_FREEBOARD_OBS = 1
}
sealand_obs_t;

struct observation {
    double lon;
    double lat;
    int type;
    double value;
    double sigma;
    double logsigma;
    int month;
    int DENSITY_ICE;
};

static std::vector < struct observation > obs;

extern "C" {

//
// This method must tell the framework how many models and how many
// hierarchical parameters are needed.
//
int gvcart_initialise_(int * nmodels,
                       int * nhierarchical) {
    * nmodels = 3; // Snow thickness and Ice thickness for CryoSat-2 and AltiKa
    * nhierarchical = 2; // CryoSat-2 and AltiKa Observations


    return 0;
}

//
// Load data and for each observation, call a callback to tell
// the inversion about the points involved
//
int gvcart_loaddata_(int * filename_len,
                     const char * filename,
                     gvcart_addobservation_t addobs) {
    int n;
    int mi[3];
    double xs[3];
    double ys[3];
    int month;

    FILE * fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "error: failed to open file %s for reading\n", filename);
        return -1;
    }

    int nobs;
    if (fscanf(fp, "%d\n", & nobs) != 1) {
        fprintf(stderr, "error: failed to read no. entries\n");
        return -1;
    }

    obs.resize(nobs);
    for (int i = 0; i < nobs; i++) {
        if (fscanf(fp, "%lf %lf %d %lf %lf %d %d \n",
            &obs[i].lon, 
            &obs[i].lat,
            &obs[i].type, 
            &obs[i].value, 
            &obs[i].sigma,
            &obs[i].month,
            &obs[i].DENSITY_ICE) != 7) {
        fprintf(stderr, "error: failed to parse observation %d/%d\n", i, nobs);
        return -1;
        }
        switch (obs[i].type) {

            case CS2_FREEBOARD_OBS:
                //n = 2;
                n=3;
                mi[0] = SNOW_MODEL;
                mi[1] = ICE_MODEL; 
                mi[2] = CS2_PENETRATION_MODEL;

                xs[0] = obs[i].lon;
                xs[1] = obs[i].lon;
                xs[2] = obs[i].lon;

                ys[0] = obs[i].lat;
                ys[1] = obs[i].lat;
                ys[2] = obs[i].lat;
                break;

            case AK_FREEBOARD_OBS:
                n=2;
                //n = 3;
                mi[0] = SNOW_MODEL;
                mi[1] = ICE_MODEL;
                //mi[2] = AK_PENETRATION_MODEL;
                
                xs[0] = obs[i].lon;
                xs[1] = obs[i].lon;
                //xs[2] = obs[i].lon;

                ys[0] = obs[i].lat;
                ys[1] = obs[i].lat;
                //ys[2] = obs[i].lat;
                break;            

            default:
                fprintf(stderr, "error: invalid type %d\n", obs[i].type);
                return -1;
        }

        if (addobs( & n, mi, xs, ys) < 0) {
            fprintf(stderr, "error: failed to add observation\n");
            return -1;
        }

        if (obs[i].sigma > 0.0) {
            obs[i].logsigma = log(obs[i].sigma);
        } else {
            fprintf(stderr, "error: invalid sigma %f\n", obs[i].sigma);
            return -1;
        }

    }

    fclose(fp);
    return 0;
}

//
// For a single observation, compute the prediction given
// model values for each point
//
int gvcart_compute_prediction_(int * nmodels,
                               int * observation,
                               int * npoints,
                               const double * value,
                               double * weight,
                               double * prediction) {
    if (( * observation) < 0 || ( * observation) >= (int) obs.size()) {
        return -1;
    }


    double DENSITY_ICE = obs[*observation].DENSITY_ICE;

    int month;
    double DENSITY_SNOW;
    month = obs[*observation].month;

    DENSITY_SNOW  = 6.5*month + 274.51;


    switch (obs[ * observation].type) {

        case CS2_FREEBOARD_OBS: // CS2 Freeboard Observations, we have two models loaded, snow->value[0] and ice->value[1]
            prediction[0] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER * value[1] + (1  - SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW*value[2] - DENSITY_SNOW/DENSITY_WATER) * value[0];
            //prediction[0] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER * value[1] + (1  - SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW - DENSITY_SNOW/DENSITY_WATER) * value[0];
            weight[0] = (1  -  DENSITY_SNOW/DENSITY_WATER);
            weight[1] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER;
            weight[2] = - SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW;

            break;

        case AK_FREEBOARD_OBS: // AK Freeboard Observations, we have  two models loaded, snow->value[0] and ice->value[1]
            prediction[0] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER * value[1] + (1 - DENSITY_SNOW/DENSITY_WATER) * value[0];
            //prediction[0] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER * value[1] + (1  - SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW*value[2] - DENSITY_SNOW/DENSITY_WATER) * value[0];
            weight[0] = (1  -  DENSITY_SNOW/DENSITY_WATER);
            weight[1] = (DENSITY_WATER - DENSITY_ICE)/DENSITY_WATER;
            weight[2] = - SPEED_OF_LIGHT_VACUUM_OVER_SPEED_OF_LIGHT_SNOW;

            break;

        default:
            return -1;
    }
    return 0;
}

//
// For the observations, given predictions, compute residuals, likelihood
// and norm
//

int gvcart_compute_likelihood_(int * nmodel,
                               int * nhierarchical,
                               int * nobservation,
                               double * hierarchical,
                               double * predictions,
                               double * residuals,
                               double * weight,
                               double * _like,
                               double * _norm)

{
    constexpr double NORMSCALE = 0.9189385332046727; //0.5*log(2.0*M_PI);
    double sum = 0.0;
    double norm = 0.0;
    double loghierarchical[3];
    for (int i = 0; i < * nhierarchical; i++) {
        loghierarchical[i] = log(hierarchical[i]);
    }

    for (int i = 0; i < ( * nobservation); i++) {
        double res = predictions[i] - obs[i].value;
        residuals[i] = res;

        double n = obs[i].sigma;
        double ln = NORMSCALE + obs[i].logsigma;

        switch (obs[i].type) {

            case CS2_FREEBOARD_OBS:
                n *= hierarchical[0];
                ln += loghierarchical[0];
                break;

            case AK_FREEBOARD_OBS:
                n *= hierarchical[1];
                ln += loghierarchical[1];
                break;

            default:
                fprintf(stderr, "gvcart_compute_likelihood: unknown observation type\n");
                return -1;
        }

        weight[i] = res / (n * n);

        sum += (res * res) / (2.0 * n * n);
        norm += ln;
    }

    * _like = sum;
    * _norm = norm;

    return 0;
}

//
// Used for making synthetic datasets, save data in correct format
// using predictions to overwrite observations
//

int gvcart_savedata_(int * n,
                     const char * filename,
                     double * noiselevel,
                     int * nobservations,
                     double * predictions) {
    FILE * fp;

    fp = fopen(filename, "w");
    if (fp == NULL) {
        return -1;
    }

    fprintf(fp, "%d\n", (int) obs.size());
    int i = 0;
    for (auto & o: obs) {
        fprintf(fp, "%16.9f %16.9f %d  %16.9f %16.9f\n",
                o.lon, o.lat, o.type, predictions[i], noiselevel[0]);
        i++;
    }

    fclose(fp);
    return 0;
}

//
// Derived variable is predicted tide gauge, i.e. Sea rate model - Land rate model
//

//
// For (optional) post processing, this function can be defined to
// compute derived mean, stddev etc from one or more models
//

double gvcart_compute_derived_(int * nmodels,
                               double * x, double * y,
                               double * values) {

    // return values[SEA_MODEL] - values[LAND_MODEL];
    // We don't *actually* have any other derived value but maybe in the future this is where we compute the
    // standard deviation??
}

}