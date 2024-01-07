#include "matrix.h"

typedef struct {
    unsigned int num_nodes;
    unsigned int num_weights;
    Mat *weights;
    double *output;
    double *error;
    double *dev_weights;
    double *dev_error;
    double *dev_output;
    double* dev_alfa_error_sig_der;
} Layer;

typedef struct{
    unsigned int num_layers;
    unsigned int added_layers;
    Layer** layer;
} Model;

Model* create_model(unsigned int num_layers);
int add_layer(Model* model, unsigned int num_nodes);
void model_summary(Model* m);
void destroy_model(Model* model);

void feed_forward(Model* model, double* train_set);
void backpropagation(Model* model, double learning_rate);
void error_backpropagation(Model* model, double* target);
void train(Model* model, double* train_set, unsigned int train_set_size, double* target, unsigned int target_size, double learning_rate);
int query(Model* model, double* input);

void mat_vect_multiplication_sequential(double *mat, double *vect,int n, int m,double *res);
__global__ void feed_forward_calc(double *device_Mat, double *device_Vect,int matRowSize, int vlength,double *device_ResVect);
void cpyWeightsDeviceToHost(Model *model);
