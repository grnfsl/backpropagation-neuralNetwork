#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nn.h>
#include <string.h>

#include <omp.h>

int main()
{
//------------------------------- Prepare training data ---------------------------------------------------
    printf("Preparing training data ... \n");
    FILE *file = fopen("mnist_dataset/mnist_train.csv", "r");
    if (file == NULL)
    {
        perror("Error while opening the file");
        exit(EXIT_FAILURE);
    }

    unsigned int i, j, rows=60000, columns=784, num_output=10;
    double** train_set = (double**)malloc(rows * sizeof (double*));

    double** target_set = (double**)malloc(rows * sizeof (double*));

    for(i = 0; i < rows; ++i)
        train_set[i] = (double*) malloc(columns * sizeof (double));

    for(i = 0; i < rows; ++i)
        target_set[i] = (double*) malloc(num_output * sizeof (double));

    //each number will have 0.99 in the index correspoding to that number and all others are 0.01
    for(i = 0; i < rows; ++i)
        for(j = 0; j < num_output; ++j)
            target_set[i][j] = 0.01;

    char img[3137];

    char *pix, *saveptr;

    for(i = 0; i < rows; ++i){
        fscanf(file, "%s", img);
        pix = strtok_r(img, ",", &saveptr);
        target_set[i][ strtol(pix, NULL, 10) ] = 0.99;
        pix = strtok_r(NULL, ",", &saveptr);
        j = 0;
        while (pix != NULL) {
            //change values from 0-255 to 0.01-1.0
            train_set[i][j] = (strtod(pix, NULL) / 255.0 * 0.99) + 0.01;
            pix = strtok_r(NULL, ",", &saveptr);
            ++j;
        }
    }
    printf("done\n");
    
//-----------------------------------------------Training neural network--------------------------------------------------------
    printf("Training ... \n");
    //neural network initialization
    Model* model = create_model(3);
    add_layer(model, columns);
    add_layer(model, 1000);
    add_layer(model, num_output);

    printf("\nNetwork summary\n");
    model_summary(model);

    double start = omp_get_wtime();
    for(i = 0; i < rows; ++i){
        train(model, train_set[i], columns, target_set[i], 10, 0.2);
    }
    double time_spent_parallel = omp_get_wtime() - start;
    printf("\nParallel time: %lf\n", time_spent_parallel);

    printf("done\n");
    printf("\nPreparing testing data ... ");

//----------------------------------------------- Prepare testing data -----------------------------------------------------------
    FILE *file_test = fopen("mnist_dataset/mnist_test.csv", "r");
    if (file == NULL)
    {
        perror("Error while opening the file");
        exit(EXIT_FAILURE);
    }

    unsigned int rows_t=10000;
    double** test_set = (double**)malloc(rows_t * sizeof (double*));

    int* target_set_t = (int*)malloc(rows_t * sizeof (int));

    for(i = 0; i < rows_t; ++i)
        test_set[i] = (double*) malloc(columns * sizeof (double));

    for(i = 0; i < rows_t; ++i){
        fscanf(file_test, "%s", img);
        pix = strtok_r(img, ",", &saveptr);
        target_set_t[i] = (int)strtol(pix, NULL, 10);
        pix = strtok_r(NULL, ",", &saveptr);
        j = 0;
        while (pix != NULL) {
            //change values from 0-255 to 0.01-1.0
            test_set[i][j] = (strtod(pix, NULL) / 255.0 * 0.99) + 0.01;
            pix = strtok_r(NULL, ",", &saveptr);
            ++j;
        }
    }
    printf("done\nTesting data ... ");

//----------------------------------------------- Test neural network -----------------------------------------------------------
    printf("Testing data ... \n");
    int network_answer;
    int currect_answer;
    int count = 0;
    for(i = 0; i < rows_t; ++i){
        network_answer = query(model, test_set[i]);
        currect_answer = target_set_t[i];
        if(network_answer == currect_answer) ++count;
    }
    printf("\nPerformance: %lf\n", count/(double)rows_t);

//-----------------------------------------------------------------------------------------------------------------------------


    fclose(file);
    fclose(file_test);

    destroy_model(model);

    for(i = 0; i < rows; ++i)
        free(train_set[i]);
    free(train_set);

    for(i = 0; i < rows; ++i)
        free(target_set[i]);
    free(target_set);

    free(target_set_t);
    for(i = 0; i < rows; ++i)
        free(test_set[i]);
    free(test_set);

    return 0;
}
