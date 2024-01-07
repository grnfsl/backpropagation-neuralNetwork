#include <matrix.h>

typedef struct {
    unsigned int num_nodes;
    unsigned int num_weights;
    Mat* weights;
    double* output;
    double* error;
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
