//
//  main.c
//  Option Pricing Tests
//  Created by Brian Savage on 3/28/20.
//  Copyright © 2020 Brian Savage. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* * * * * * * * * * * * *
 *                       *
 *        Defines        *
 *                       *
 * * * * * * * * * * * * */
#define GBM_WEIGHT 0.775                     // All of these weights must sum up to 1
#define JUMP_DIFF_WEIGHT (1-GBM_WEIGHT)      // a later edition may compute these weights for you
#define SQUARE_ROOT_WEIGHT 0.000             // based off market conditions and indictators to which model fits best

#define INDEPENDENT_SIMULATION_ITERATIONS 50 // Number of independent simulation iterations
#define NUMBER_OF_SIMULATIONS 50000          // Number of Monte Carlo Type Simulations
#define NUMBER_CONTRACTS_ABOVE 20            // Number of option contracts above closest contract
#define NUMBER_CONTRACTS_BELOW 20            // Number of option contracts below closest contract

// 2020 trading year
// 1=trading day, 0=not   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
bool trading_days[365] = {0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                          0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0,
                          0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                          1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                          1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                          1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                          1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                          0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                          1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
                          0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
                          1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1};

/* * * * * * * * * * * * *
*                       *
*        Structs        *
*                       *
* * * * * * * * * * * * */
typedef struct Today {
    unsigned month;
    unsigned   day;
    unsigned  year;
} Date;

typedef struct Metrics {
    float      open;
    float      high;
    float       low;
    float     close;
    float adj_close;
    double   volume;
} Metrics;

typedef struct Daily {
    struct Today       *date;
    struct Metrics    *ohlcv;
    struct Daily   *next_day;
    struct Daily   *prev_day;
} Daily;
 
typedef struct Historical {
    char *symbol;
    unsigned lookback_period;
    struct Daily *start;
    struct Daily   *end;
} Historical;

struct GBM_Node {
    float price;           // predicted price
    float rsv;             // random shock value
    float crsv;            // Brownian cumulative random shock value
    float drift;           // asset drift weight
    float diffusion;       // asset diffusion rate
    struct GBM_Node *next; // next prediction node
} GBM_Node;

struct Jump_Diff_Node {
    float price;                 // predicted price
    float rsv;                   // random shock value
    float crsv;                  // Brownian cumulative random shock value
    float jump_magnitude;        // normal random variable for jump magnitude
    float jump_timing;           // Poisson Distribution Random Variable for jumps timing
    float mean_magnitude;        // mean jump magnitude
    float drift_correction;      // asset drift weight
    struct Jump_Diff_Node *next; // next prediction node
} Jump_Diff_Node;

/* * * * * * * * * * * * *
 *                       *
 *      Declarations     *
 *                       *
 * * * * * * * * * * * * */
void initialize_asset(struct Historical *stock, char *symbol, char *filename);
void initialize_daily(struct Daily *stock, FILE *fp);
bool validate_columns(FILE *fp);
void recycle_asset_allocated_memory(struct Historical *asset);

float randflt(void);
float random_poisson(float lamda);
float random_box_muller_transform(void);

unsigned days_till_expiration(unsigned last_month, unsigned last_day, unsigned last_year);
float ohlc_avg(struct Metrics *day);
float compute_mu(struct Daily *start_lookback, unsigned option_exp);
float compute_sigma(struct Daily *start_lookback, float mu, unsigned option_exp);
float compute_cumulative_sum(struct GBM_Node *start, unsigned days_since_start);
float cumulative_sum_jump_diff(struct Jump_Diff_Node *start, unsigned days_since_start);
struct Daily *get_day(struct Historical *stock, unsigned lookback);

float Geometric_Brownian_Motion(struct Daily *lookback_start, float start_price, unsigned option_exp);
float Jump_Diffusion(struct Daily *lookback_start, float start_price, unsigned option_exp);
float Square_Root_Diffusion(struct Daily *lookback_start, float start_price, unsigned option_exp);
void predicted_option_values(struct Historical *stock, unsigned option_exp);

/* * * * * * * * * * * * *
 *                       *
 *         Main          *
 *                       *
 * * * * * * * * * * * * */
int main(int argc, char *argv[]) {
    assert(argc > 1);
    srand((unsigned)time(NULL));
    for (unsigned i = 1; i < argc; i++) {
        struct Historical *stock = malloc(sizeof(struct Historical));
        assert(stock != NULL);
        char buff[10];
        strcpy(buff, argv[i]);
        initialize_asset(stock, strtok(&buff[0], "."), argv[i]);
        printf("+-----------+ Predicted %s Option Values +----------+\n\n", stock->symbol);
        predicted_option_values(stock, days_till_expiration(stock->end->date->month, stock->end->date->day, stock->end->date->year););
        printf("\n+----------------------------------------------------+\n\n\n");
        recycle_asset_allocated_memory(stock);
    }
    return EXIT_SUCCESS;
}

/* * * * * * * * * * * * *
 *                       *
 *       Functions       *
 *                       *
 * * * * * * * * * * * * */
void predicted_option_values(struct Historical *stock, unsigned option_exp)
{
    assert(stock != NULL);
    assert(stock->lookback_period >= option_exp);
    
    double GBM_pred_err = 0;
    double Jump_Diff_pred_err = 0;
    double Square_Root_pred_err = 0;
    for (unsigned i = 1; i < INDEPENDENT_SIMULATION_ITERATIONS; i++) {
        GBM_pred_err += Geometric_Brownian_Motion(get_day(stock, option_exp * 2), get_day(stock, option_exp)->ohlcv->close, option_exp);
        Jump_Diff_pred_err += Jump_Diffusion(get_day(stock, option_exp * 2), get_day(stock, option_exp)->ohlcv->close, option_exp);
        Square_Root_pred_err += Square_Root_Diffusion(get_day(stock, option_exp * 2), get_day(stock, option_exp)->ohlcv->close, option_exp);
    }
    GBM_pred_err /= INDEPENDENT_SIMULATION_ITERATIONS;
    Jump_Diff_pred_err /= INDEPENDENT_SIMULATION_ITERATIONS;
    Square_Root_pred_err /= INDEPENDENT_SIMULATION_ITERATIONS;
    float pred_err = (float)(GBM_pred_err * GBM_WEIGHT) + (Jump_Diff_pred_err * JUMP_DIFF_WEIGHT) + (Square_Root_pred_err * SQUARE_ROOT_WEIGHT);
    float model_previous_percent_error = ((pred_err - get_day(stock, option_exp)->ohlcv->close) / get_day(stock, option_exp)->ohlcv->close) * 100;
    if (model_previous_percent_error > 0) {
        printf("\n    Previous %% Error: [+%0.2f%%] Model Over Estimated\n", model_previous_percent_error);
    } else {
        printf("\n    Previous %% Error: [%0.2f%%] Model Under Estimated\n", model_previous_percent_error);
    }
    
    double GBM_pred = 0;
    double Jump_Diff_pred = 0;
    double Square_Root_pred = 0;
    for (unsigned i = 1; i < INDEPENDENT_SIMULATION_ITERATIONS; i++) {
        GBM_pred += Geometric_Brownian_Motion(get_day(stock, option_exp), stock->end->ohlcv->close, option_exp);
        Jump_Diff_pred += Jump_Diffusion(get_day(stock, option_exp), stock->end->ohlcv->close, option_exp);
        Square_Root_pred += Square_Root_Diffusion(get_day(stock, option_exp), stock->end->ohlcv->close, option_exp);
    }
    GBM_pred /= INDEPENDENT_SIMULATION_ITERATIONS;
    Jump_Diff_pred /= INDEPENDENT_SIMULATION_ITERATIONS;
    Square_Root_pred /= INDEPENDENT_SIMULATION_ITERATIONS;
    
    float predicted_value = (float)(GBM_pred * GBM_WEIGHT) + (Jump_Diff_pred * JUMP_DIFF_WEIGHT) + (Square_Root_pred * SQUARE_ROOT_WEIGHT);
    printf("\n                    Current Price: %0.2f\n\n", stock->end->ohlcv->close);
    printf("          Average Predicted Price: %0.2f   Weights \n", predicted_value);
    printf("  Geometric Brownian Motion Price: %0.2f  [ %0.3f ]\n", GBM_pred, GBM_WEIGHT);
    printf("             Jump Diffusion Price: %0.2f  [ %0.3f ]\n", Jump_Diff_pred, JUMP_DIFF_WEIGHT);
    printf("      Square Root Diffusion Price: %0.2f   [ %0.3f ]\n\n", Square_Root_pred, SQUARE_ROOT_WEIGHT);
    printf("            CALL                      PUT           \n   Strike   -|->   $ Est     Strike   -|->   $ Est\n");
    for (float strike_price = round(stock->end->ohlcv->close) + (NUMBER_CONTRACTS_ABOVE / 2); strike_price >= round(stock->end->ohlcv->close) - (NUMBER_CONTRACTS_BELOW / 2); strike_price -= 0.5) {
        if (round(stock->end->ohlcv->close) == strike_price) {
            if (predicted_value - strike_price > 0) {
                printf("\n-> [%0.2f] --|--> [+%0.2f] <", strike_price, predicted_value - strike_price);
                printf("> [%0.2f] --|--> [ -0- ] <-\n\n", strike_price);
            } else {
                printf("\n-> [%0.2f] --|--> [ -0- ] <", strike_price);
                printf("> [%0.2f] --|--> [+%0.2f] <-\n\n", strike_price, strike_price - predicted_value);
            }
        } else {
            if (predicted_value - strike_price > 0) {
                printf("   [%0.2f] --|--> [+%0.2f] ", strike_price, predicted_value - strike_price);
                printf("   [%0.2f] --|--> [ -0- ]\n", strike_price);
            } else {
                printf("   [%0.2f] --|--> [ -0- ] ", strike_price);
                printf("   [%0.2f] --|--> [+%0.2f]\n", strike_price, strike_price - predicted_value);
            }
        }
    }
}

float Geometric_Brownian_Motion(struct Daily *lookback_start, float start_price, unsigned option_exp)
{
    assert(lookback_start != NULL);
    /*
     *  Steps to Compute Avg Final Price At Maturity
     *    (1) Check option maturity does exceed lookback length
     *    (2) Parameters to GBM Simulation
     *           ip    : initial price
     *           dt    : time incremenet
     *           exp   : length till option expiraton
     *           mu    : mean of historical daily returns
     *           sigma : standard deviation of historical daily returns (another potential value for sigma is the implied volatility of the option)
     *           b     : array for brownian increments
     *           w     : array for brownian path
     */
    float cp = start_price;
    float mu = compute_mu(lookback_start, option_exp); // mu is the mean return of the stock prices within the historical lookback of the time till option exp
    float sigma = compute_sigma(lookback_start, mu, option_exp); // sigma is the standard deviation of returns
    
    // Initialization of Pricing Simulation
    struct GBM_Node *simulations[NUMBER_OF_SIMULATIONS];
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        simulations[i] = malloc(sizeof(GBM_Node));
        assert(simulations[i] != NULL);
        simulations[i]->rsv = random_box_muller_transform();
        simulations[i]->crsv = 0;
        simulations[i]->price = 0;
        simulations[i]->drift = 0;
        simulations[i]->diffusion = 0;
        struct GBM_Node *curr = simulations[i]->next;
        struct GBM_Node *prev = simulations[i];
        for (unsigned j = 1; j < option_exp; j++) {
            curr = malloc(sizeof(GBM_Node));
            assert(curr != NULL);
            curr->rsv = random_box_muller_transform();
            curr->crsv = 0;
            curr->price = 0;
            curr->drift = 0;
            curr->diffusion = 0;
            prev->next = curr;
            prev = prev->next;
            curr = curr->next;
        }
        prev->next = NULL;
    }
    // Computing Cumulative Sum of Random Shocks
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        unsigned day_count = 1;
        struct GBM_Node *curr_node = simulations[i];
        while (curr_node != NULL) {
            assert(curr_node != NULL);
            curr_node->crsv = compute_cumulative_sum(simulations[i], day_count);
            day_count++;
            curr_node = curr_node->next;
        }
    }
    // Computing Average Maturity Price of Asset
    float avg_expiration_price = 0;
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        struct GBM_Node *curr_node = simulations[i];
        for (unsigned j = 0; j < option_exp; j++) {
            curr_node->drift = (mu - (0.5 * pow(sigma, 2))) * j;
            curr_node->diffusion = sigma * curr_node->crsv;
            curr_node->price = cp * exp(curr_node->drift + curr_node->diffusion);
            if (j < option_exp - 1) {
                curr_node = curr_node->next;
            } else {
                avg_expiration_price += curr_node->price;
            }
        }
    }
    // Deallocating memory used for each simulation
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        struct GBM_Node *node = simulations[i];
        struct GBM_Node *next = NULL;
        while (node != NULL) {
            next = node->next;
            free(node);
            node = next;
        }
    }
    // Returning the final average expiration price
    return avg_expiration_price / NUMBER_OF_SIMULATIONS;
}

float Jump_Diffusion(struct Daily *lookback_start, float start_price, unsigned option_exp)
{
    assert(lookback_start != NULL);

    float cp = start_price;                                      // cp is the most recent real close price
    float mu = compute_mu(lookback_start, option_exp);           // mu is the mean return of the stock prices within the historical lookback of the time till option exp
    float sigma = compute_sigma(lookback_start, mu, option_exp); // sigma is the standard deviation of returns
    float lamda = 0.75;                                          // lamda is the jump intensity constant
    float delta = 0.25;                                          // delta tbh no idea but it was in the model
    float risk_free = 0.001;                                      // risk free rate
    
    // Initialization of Pricing Simulation
    struct Jump_Diff_Node *simulations[NUMBER_OF_SIMULATIONS];
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        simulations[i] = malloc(sizeof(Jump_Diff_Node));
        assert(simulations[i] != NULL);
        simulations[i]->rsv = random_box_muller_transform();
        simulations[i]->jump_magnitude = random_box_muller_transform();
        simulations[i]->jump_timing = random_poisson(lamda);
        simulations[i]->mean_magnitude = 0;
        simulations[i]->crsv = 0;
        simulations[i]->price = 0;
        simulations[i]->drift_correction = 0;
        struct Jump_Diff_Node *curr = simulations[i]->next;
        struct Jump_Diff_Node *prev = simulations[i];
        for (unsigned j = 1; j < option_exp; j++) {
            curr = malloc(sizeof(Jump_Diff_Node));
            assert(curr != NULL);
            curr->rsv = random_box_muller_transform();
            curr->jump_magnitude = random_box_muller_transform();
            curr->jump_timing = random_poisson(lamda);
            curr->mean_magnitude = 0;
            curr->crsv = 0;
            curr->price = 0;
            curr->drift_correction = 0;
            prev->next = curr;
            prev = prev->next;
            curr = curr->next;
        }
        prev->next = NULL;
    }
    
    // Computing Mean Jump Size
    float mean_jump_size;
    unsigned days;
    struct Jump_Diff_Node *curr = NULL;
    for (unsigned i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        days = 0;
        mean_jump_size = 0;
        curr = simulations[i];
        while (curr != NULL) {
            assert(curr != NULL);
            mean_jump_size += curr->jump_magnitude;
            days++;
            curr = curr->next;
        }
        mean_jump_size /= days;
        curr = simulations[i];
        while (curr != NULL) {
            assert(curr != NULL);
            curr->mean_magnitude = mean_jump_size;
            curr = curr->next;
        }
    }
    
    // Computing Cumulative Sum of Random Shocks
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        unsigned days = 1;
        struct Jump_Diff_Node *curr = simulations[i];
        while (curr != NULL) {
            assert(curr != NULL);
            curr->crsv = cumulative_sum_jump_diff(simulations[i], days);
            days++;
            curr = curr->next;
        }
    }
    
    // Computing Drift Correction to maintain the risk-neutral measure
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        struct Jump_Diff_Node *curr = simulations[i];
        while (curr != NULL) {
            assert(curr != NULL);
            curr->drift_correction = lamda * exp(curr->mean_magnitude + ((delta * delta) / 2) - 1);
            curr = curr->next;
        }
    }
    
    float rolling_avg = 0;
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        struct Jump_Diff_Node *curr = simulations[i];
        unsigned days = 1;
        while (curr != NULL) {
            assert(curr != NULL);
            curr->price = cp * (exp((risk_free - curr->drift_correction - ((sigma * sigma) / 2)) + (sigma * curr->crsv)) +
                               ((exp(curr->mean_magnitude + (delta * curr->jump_magnitude)) - 1) * curr->jump_timing));
            if (curr->next == NULL) {
                rolling_avg += curr->price;
            }
            curr = curr->next;
            days++;
        }
    }
    
    // Deallocating memory used for each simulation
    for (int i = 0; i < NUMBER_OF_SIMULATIONS; i++) {
        struct Jump_Diff_Node *node = simulations[i];
        struct Jump_Diff_Node *next = NULL;
        while (node != NULL) {
            next = node->next;
            free(node);
            node = next;
        }
    }
    // Returning the final average expiration price
    return rolling_avg / NUMBER_OF_SIMULATIONS;
}

float Square_Root_Diffusion(struct Daily *lookback_start, float start_price, unsigned option_exp)
{
    assert(lookback_start != NULL);

    return 0.0;
}

/* * * * * * * * * * * * *
 *                       *
 *      Helper Funcs     *
 *                       *
 * * * * * * * * * * * * */
void initialize_asset(struct Historical *stock, char *symbol, char *filename)
{
    FILE *fp = fopen(filename, "r");
    assert(fp != NULL);
    assert(validate_columns(fp) == true);
    
    stock->symbol = symbol;
    stock->lookback_period = 0;
    struct Daily *curr = stock->start;
    struct Daily *prev = NULL;
    
    do {
        curr = malloc(sizeof(Daily));
        assert(curr != NULL);
        initialize_daily(curr, fp);
        curr->prev_day = prev;
        if (prev != NULL) {
            prev->next_day = curr;
        } else {
            stock->start = curr;
        }
        prev = curr;
        stock->lookback_period++;
    } while (!feof(fp));
    
    curr->next_day = NULL;
    stock->end = curr;
    fclose(fp);
}


void initialize_daily(struct Daily *stock, FILE *fp)
{
    assert(stock != NULL);
    assert(fp    != NULL);
    stock->ohlcv = malloc(sizeof(Metrics));
    assert(stock->ohlcv != NULL);
    stock->date = malloc(sizeof(Date));
    assert(stock->date != NULL);
    assert(fscanf(fp, "%u-%u-%u,%f,%f,%f,%f,%f,%lf\n", &stock->date->year,
                                                       &stock->date->month,
                                                       &stock->date->day,
                                                       &stock->ohlcv->open,
                                                       &stock->ohlcv->high,
                                                       &stock->ohlcv->low,
                                                       &stock->ohlcv->close,
                                                       &stock->ohlcv->adj_close,
                                                       &stock->ohlcv->volume) == 9);
}

bool validate_columns(FILE *fp) {
    assert(fp != NULL);
    char *before = malloc(sizeof(char *));
    assert(before != NULL);
    char *after  = malloc(sizeof(char *));
    assert(after != NULL);
    bool is_correct_formatting = (bool) fscanf(fp, "%s %s\n", before, after);
    free(before);
    free(after);
    return is_correct_formatting;
}

void recycle_asset_allocated_memory(struct Historical *asset)
{
    assert(asset != NULL);
    struct Daily *current_day = asset->start;
    struct Daily *next_day;
    while (current_day != NULL) {
        next_day = current_day->next_day;
        free(current_day->date);
        free(current_day->ohlcv);
        free(current_day);
        current_day = next_day;
    }
    free(asset);
}

float ohlc_avg(struct Metrics *day)
{
    assert(day != NULL);
    return ((day->open + day->high + day->low + day->close) / 4);
}

float compute_mu(struct Daily *start_lookback, unsigned option_exp)
{
    assert(start_lookback != NULL);
    float rolling_returns = 0.0;
    struct Daily *current = start_lookback;
    for (unsigned i = 0; i < option_exp; i++) {
        rolling_returns += ((current->ohlcv->close - current->prev_day->ohlcv->close) / (current->prev_day->ohlcv->close));
        current = current->next_day;
    }
    return rolling_returns / option_exp;
}

float compute_sigma(struct Daily *start_lookback, float mu, unsigned option_exp)
{
    assert(start_lookback != NULL);
    float diff_from_mu = 0.0;
    float daily_returns = 0.0;
    struct Daily *current = start_lookback;
    for (unsigned i = 0; i < option_exp; i++) {
        daily_returns = ((current->ohlcv->close - current->prev_day->ohlcv->close) / (current->prev_day->ohlcv->close));
        diff_from_mu += pow(daily_returns - mu, 2);
        current = current->next_day;
    }
    return sqrt(diff_from_mu / option_exp);
}

float compute_cumulative_sum(struct GBM_Node *start, unsigned days_since_start)
{
    assert(start != NULL);
    if (days_since_start == 0)
        return start->rsv;
    float rolling_sum = 0.0;
    struct GBM_Node *curr_node = start;
    for (unsigned i = 0; i < days_since_start - 1; i++) {
        rolling_sum += curr_node->rsv;
        curr_node = curr_node->next;
    }
    return rolling_sum;
}

float cumulative_sum_jump_diff(struct Jump_Diff_Node *start, unsigned days_since_start)
{
    assert(start != NULL);
    if (days_since_start == 0)
        return start->rsv;
    float rolling_sum = 0.0;
    struct Jump_Diff_Node *curr = start;
    for (unsigned i = 0; i < days_since_start - 1; i++) {
        rolling_sum += curr->rsv;
        curr = curr->next;
    }
    return rolling_sum;
    
    
}

float random_poisson(float lamda)
{
    float rand = 0;
    float rolling = 1;
    float exp_of_neg_lamda = exp(-lamda);
    do {
        ++rand;
        rolling *= randflt();
    } while (rolling > exp_of_neg_lamda);
    return --rand;
}

float random_box_muller_transform(void)
{
    float x1, x2, w;
    do {
        x1 = 2. * randflt() - 1.;
        x2 = 2. * randflt() - 1.;
        w = (x2 * x2) + (x1 * x1);
    } while (1 <= w);
    return x1 * (sqrt(-2. * log(w) / w));
}

float randflt(void)
{
    return ((float)rand() / (float)(RAND_MAX));
}


struct Daily *get_day(struct Historical *stock, unsigned lookback)
{
    assert(stock != NULL);
    struct Daily *current_day = stock->end;
    for (unsigned i = 0; i < lookback; i++) {
        current_day = current_day->prev_day;
    }
    assert(current_day != NULL);
    return current_day;
}

unsigned days_till_expiration(unsigned last_month, unsigned last_day, unsigned last_year)
{
    unsigned month, day, year;
    unsigned days_in_month[12] = {31, 28, 31,
                                  30, 31, 30,
                                  31, 31, 30,
                                  31, 30, 31};
    printf("     EXPIRATION DATE [MM/DD/YYYY]: ");
    scanf("%u/%u/%u", &month, &day, &year);
    unsigned start = 0;
    for (unsigned i = 1; i < last_month; i++) {
        start += days_in_month[i - 1];
    }
    start += last_day;
    unsigned end = 0;
    for (unsigned i = 1; i < month; i++) {
        end += days_in_month[i - 1];
    }
    end += day;
    unsigned trading_day_count = 0;
    trading_day_count = (year - last_year) * 252;
    for (unsigned i = start; i < end; i++) {
        if (trading_days[i])
            trading_day_count++;
    }
    printf("     Trading days till expiration: %u days\n", trading_day_count);
    return trading_day_count;
}
