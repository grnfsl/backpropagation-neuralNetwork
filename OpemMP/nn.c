#include <nn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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
    }
    else{
        model->layer[model->added_layers]->num_nodes = num_nodes;
        model->layer[model->added_layers]->num_weights = num_nodes * model->layer[model->added_layers-1]->num_nodes;
        double range = 1/sqrt(model->layer[model->added_layers-1]->num_nodes);
        model->layer[model->added_layers]->weights = create_random_mat(model->layer[model->added_layers]->num_nodes, model->layer[model->added_layers-1]->num_nodes, -range, range);
//        sleep(1);
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

void feed_forward(Model* model, double* train_set)
{
    model->layer[0]->output = train_set;
    double* tmp;
    unsigned int i;
    for(i = 1; i < model->num_layers; ++i){
        tmp = (double*)malloc(model->layer[i]->weights->n * sizeof (double));
        mat_vect_mult(model->layer[i]->weights, model->layer[i-1]->output, model->layer[i-1]->num_nodes, tmp, NUM_THREADS);
        model->layer[i]->output = vect_sigmoid(model->layer[i]->num_nodes, tmp, NUM_THREADS);
        free(tmp);
    }
}

void backpropagation(Model* model, double learning_rate)
{
//    printf("\n------------------------------Backpropagation------------------------------------\n");
    unsigned int k;
//    omp_set_nested(1);
//#pragma omp parallel for num_threads(NUM_THREADS)
    for(k = model->num_layers-1; k > 0; --k){
        double* alfa_error_sig_der = create_vector(model->layer[k]->num_nodes);
        unsigned int i, size = model->layer[k]->num_nodes;

        //calculate derivative of sigmoid functions and multiply by learning rate
#pragma omp parallel num_threads(NUM_THREADS)
        {
            #pragma omp for
            for(i = 0; i < size; ++i)
                alfa_error_sig_der[i] = learning_rate * (model->layer[k]->error[i]) * (model->layer[k]->output[i] * (1.0 - model->layer[k]->output[i]));

            unsigned int j, rows, columns;
            double delta;
            rows = model->layer[k]->num_nodes;
            columns = model->layer[k-1]->num_nodes;
            #pragma omp for
            for(i = 0; i < rows; ++i){
                for(j = 0; j < columns; ++j){
                    delta = (alfa_error_sig_der[i] *  model->layer[k-1]->output[j]);
                    model->layer[k]->weights->mat[i*columns + j] += delta;
                }
            }
        }
        free(alfa_error_sig_der);
    }
}

void error_backpropagation(Model* model, double* target)
{
//    printf("\n----------------------------------Errors--------------------------------\n");

    model->layer[model->num_layers-1]->error = (double*)malloc(model->layer[model->num_layers-1]->num_nodes * sizeof(double));
    vect_vect_sub(target, model->layer[model->num_layers-1]->output, model->layer[model->num_layers-1]->num_nodes, model->layer[model->num_layers-1]->error, NUM_THREADS);

    unsigned int i;
    for(i = model->num_layers-2; i > 0; --i){
        Mat* mat_T = create_mat(model->layer[i+1]->weights->m, model->layer[i+1]->weights->n);
        unsigned int k, j;
        model->layer[i]->error = (double*)malloc(mat_T->n * sizeof (double));
        unsigned int x_size = model->layer[i+1]->num_nodes;

#pragma omp parallel num_threads(1) private(k, j)
        {
            //calculate transpose
#pragma omp for
            for(k = 0; k < model->layer[i+1]->weights->n; ++k)
                for(j = 0; j < mat_T->n; ++j)
                    mat_T->mat[j*mat_T->m+k] = model->layer[i+1]->weights->mat[k*model->layer[i+1]->weights->m+j];
#pragma omp for
            for(k = 0; k < mat_T->n; ++k){
                double tmp = 0.0;
                for(j = 0; j < x_size; ++j)
                    tmp += mat_T->mat[k*x_size + j] * model->layer[i+1]->error[j];
                model->layer[i]->error[k] = tmp;
            }

        }
            destroy_mat(mat_T);

        }
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
    feed_forward(model, input);

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
            free(model->layer[i]->output);
            free(model->layer[i]->error);
        }
        free(model->layer[i]);
    }
    free(model->layer);
    free(model);
}
