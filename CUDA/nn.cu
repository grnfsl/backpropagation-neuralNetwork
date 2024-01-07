#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_THREADS 8

Model* create_model(unsigned int num_layers)
{
    Model* model = (Model*) malloc(sizeof (Model));
    Layer** layer = (Layer**) malloc(num_layers * sizeof (Layer*));
    model->layer = layer;
    for(unsigned int i = 0; i < num_layers; ++i)
        model->layer[i] = (Layer*)malloc(sizeof (Layer));
    model->num_layers = num_layers;
    model->added_layers = 0;
    return model;
}

int add_layer(Model* model, unsigned int num_nodes)
{
    if(model->added_layers >= model->num_layers){    //if try to add more than the one that seted to model
        printf("Warning: the layer is not added because the number of layers exceeded the number set by model\n");
        return 0;
    }

    if(model->added_layers == 0){
        model->layer[model->added_layers]->num_nodes = num_nodes;
        model->layer[model->added_layers]->num_weights = 0;
        model->layer[model->added_layers]->weights = create_mat(1,1);

        cudaMalloc((void**)&model->layer[model->added_layers]->dev_output, model->layer[model->added_layers]->num_nodes * sizeof (double));
    }
    else{
        model->layer[model->added_layers]->num_nodes = num_nodes;
        model->layer[model->added_layers]->num_weights = num_nodes * model->layer[model->added_layers-1]->num_nodes;
        double range = 1/sqrt(model->layer[model->added_layers-1]->num_nodes);
        model->layer[model->added_layers]->weights = create_random_mat(model->layer[model->added_layers]->num_nodes, model->layer[model->added_layers-1]->num_nodes, -range, range);
        model->layer[model->added_layers]->output = (double*)malloc(model->layer[model->added_layers]->num_nodes * sizeof (double));

        model->layer[model->added_layers]->error = (double*)malloc(model->layer[model->added_layers]->num_nodes * sizeof (double));

        cudaMalloc((void**)&model->layer[model->added_layers]->dev_weights, model->layer[model->added_layers]->weights->n * model->layer[model->added_layers]->weights->m * sizeof (double));
        cudaMemcpy(model->layer[model->added_layers]->dev_weights, model->layer[model->added_layers]->weights->mat, model->layer[model->added_layers]->weights->n * model->layer[model->added_layers]->weights->m * sizeof (double), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&model->layer[model->added_layers]->dev_output, model->layer[model->added_layers]->num_nodes * sizeof (double));     //device output vector
        cudaMalloc((void**)&model->layer[model->added_layers]->dev_error, model->layer[model->added_layers]->num_nodes * sizeof (double));      //device error vector

        cudaMalloc((void**)&model->layer[model->added_layers]->dev_alfa_error_sig_der, model->layer[model->added_layers]->num_nodes * sizeof (double));
    }
    model->added_layers += 1;

    return 1;
}

void model_summary(Model* model)
{
    if(model->num_layers != model->added_layers)
        printf("Error: the number of layers set is not equal to the number of added layers\n");

    unsigned int i;
    printf("Layer\tNodes\tParam\n");
    for(i = 0; i < model->num_layers; ++i)
        printf("dense_%d\t%u\t%u\n", i, model->layer[i]->num_nodes, model->layer[i]->num_weights);
}

void mat_vect_multiplication_sequential(double *mat, double *vect,int n, int m,double *res)
{
    for(int i=0;i<n;i++)
    {
        res[i]=0;
        for(int j = 0; j < m; j++)
            res[i] += mat[i*n+j]*vect[j];
    }
}

__global__ void feed_forward_calc(double *device_Mat, double *device_Vect,int rows, int vlength, int output_size, double *output)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < rows) {
        int i, m = tid * vlength;
        output[tid] = 0.00;
        for(i = 0; i < vlength; i++)
            output[tid] += device_Mat[m+i] * device_Vect[i];
        output[tid] = 1.0 / (1.0 + __expf(- output[tid]));
        tid += blockDim.x * gridDim.x;
    }
}

void feed(Model* model, double* input)
{
    cudaMemcpy(model->layer[0]->dev_output, input, model->layer[0]->num_nodes * sizeof (double), cudaMemcpyHostToDevice);
    unsigned int i;
    for(i = 1; i < model->num_layers; ++i){
        feed_forward_calc<<<32, 1024>>>(model->layer[i]->dev_weights, model->layer[i-1]->dev_output, model->layer[i]->weights->n, model->layer[i-1]->num_nodes, model->layer[i]->num_nodes, model->layer[i]->dev_output);
        cudaDeviceSynchronize();
        cudaMemcpy(model->layer[i]->output, model->layer[i]->dev_output, model->layer[i]->num_nodes * sizeof (double), cudaMemcpyDeviceToHost);
    }
}

void feed_forward(Model* model, double* train_set)
{
    cudaMemcpy(model->layer[0]->dev_output, train_set, model->layer[0]->num_nodes * sizeof (double), cudaMemcpyHostToDevice);
    unsigned int i;
    for(i = 1; i < model->num_layers; ++i){
        feed_forward_calc<<<32, 1024>>>(model->layer[i]->dev_weights, model->layer[i-1]->dev_output, model->layer[i]->weights->n, model->layer[i-1]->num_nodes, model->layer[i]->num_nodes, model->layer[i]->dev_output);
        cudaDeviceSynchronize();
    }
}

__global__ void backpropagation_calc(double *alfa_error_sig_der, double *weights, double *error, double *output_current, double *output_prev, int rows, int columns, double learning_rate)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < rows) {
        alfa_error_sig_der[tid] = learning_rate * (error[tid]) * (output_current[tid] * (1.0 - output_current[tid]));
        tid += blockDim.x * gridDim.x;
    }

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < rows) {
        for(int j = 0; j < columns; ++j)
            weights[tid * columns + j] += alfa_error_sig_der[tid] *  output_prev[j];
        tid += blockDim.x * gridDim.x;
    }
}

void backpropagation(Model* model, double learning_rate)
{
    for(int k = model->num_layers-1; k > 0; --k)
        backpropagation_calc<<<32, 1024>>>(model->layer[k]->dev_alfa_error_sig_der, model->layer[k]->dev_weights,model->layer[k]->dev_error, model->layer[k]->dev_output, model->layer[k-1]->dev_output, model->layer[k]->num_nodes, model->layer[k-1]->num_nodes, learning_rate);
}

__global__ void vect_vect_sub_gpu( double* target, double *output_vect, double *error_current, int target_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < target_size) {
        error_current[tid] = target[tid] - output_vect[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void error_backpropagation_calc(double *weights, int weight_n, int weight_m, double *error_prev, double *error_current)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < weight_m) {
        double tmp = 0.0;
        for(int j = 0; j < weight_n; ++j)
            tmp += weights[j * weight_m + tid] * error_current[j];
        error_prev[tid] = tmp;
        tid += blockDim.x * gridDim.x;
    }
}

void error_backpropagation(Model* model, double* target)
{
    double *dev_target;
    cudaMalloc((void**)&dev_target, model->layer[model->num_layers-1]->num_nodes * sizeof(double));
    cudaMemcpy(dev_target, target, model->layer[model->num_layers-1]->num_nodes * sizeof (double), cudaMemcpyHostToDevice);
    vect_vect_sub_gpu<<<1, 10>>>(dev_target, model->layer[model->num_layers-1]->dev_output, model->layer[model->num_layers-1]->dev_error, model->layer[model->num_layers-1]->num_nodes);
    cudaFree(dev_target);
    cudaDeviceSynchronize();

    for(int i = model->num_layers-1; i > 1; --i){
        error_backpropagation_calc<<<32, 1024>>>(model->layer[i]->dev_weights, model->layer[i]->weights->n, model->layer[i]->weights->m, model->layer[i-1]->dev_error, model->layer[i]->dev_error);
        cudaDeviceSynchronize();
    }
}

void cpyWeightsDeviceToHost(Model *model)
{
    for(int i = 1; i < model->num_layers; ++i)
        cudaMemcpy(model->layer[i]->weights->mat, model->layer[i]->dev_weights, model->layer[i]->weights->n * model->layer[i]->weights->m * sizeof (double), cudaMemcpyDeviceToHost);
}

void train(Model* model, double* train_set, unsigned int train_set_size, double* target, unsigned int target_size, double learning_rate)
{
    if(model->num_layers != model->added_layers){
        printf("Error: the number of layers set is not equal to the number of added layers\n");
        return;
    }

    if(train_set_size != model->layer[0]->num_nodes){
        printf("%u %u\n", train_set_size, model->layer[0]->num_nodes);
        printf("Error: training set values do not fit to the input layer\n");
        return;
    }

    if(target_size != model->layer[model->num_layers-1]->num_nodes){
        printf("Error: target values do not fit to the output layer\n");
        return;
    }
    
    feed_forward(model, train_set);
    error_backpropagation(model, target);
    backpropagation(model, learning_rate);
}

int query(Model* model, double* input)
{
    feed(model, input);

    double max = model->layer[model->num_layers-1]->output[0];
    int num = 0;

    unsigned int i, size = model->layer[model->num_layers-1]->num_nodes;
    for(i = 1; i < size; ++i){
        if(model->layer[model->num_layers-1]->output[i] > max){
            max = model->layer[model->num_layers-1]->output[i];
            num = (int)i;
        }
    }
    return num;
}

void destroy_model(Model* model)
{
    unsigned int i;
    for(i = 0; i < model->num_layers; ++i){
        destroy_mat(model->layer[i]->weights);
        if(i!=0){
            cudaFree(model->layer[i]->dev_weights);
            cudaFree(model->layer[i]->dev_error);
            cudaFree(model->layer[i]->dev_alfa_error_sig_der);
            free(model->layer[i]->error);
            free(model->layer[i]->output);
        }
        cudaFree(model->layer[i]->output);
//        free(model->layer[i]->output);
        free(model->layer[i]);
    }
    free(model->layer);
    free(model);
}
