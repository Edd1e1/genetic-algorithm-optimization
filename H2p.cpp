#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <random>
#include <functional>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <fstream>

using namespace std;

// =========================== PARAPEMTERS =====================================
const double PI = 3.14159265358979323846;
const int POP_SIZE = 100;
const int MAX_GENERATIONS = 1000; 
const int NUM_RUNS = 30; 
const int ELITE_SIZE = 5; 
const uint32_t GRAY_PRECISION = 4294967295; 

random_device rd;
mt19937 rng(rd());
uniform_real_distribution<double> dist01(0.0, 1.0);

// =========================== BENCHMARK FUNCTIONS ===============================

double rastrigin(const vector<double>& X){
    double sum = 10 * X.size();
    for(const auto& x : X) sum += x * x - 10 * cos(2 * PI * x);
    return sum;
}

double deJong(const vector<double>& X){
    double sum = 0;
    for(const auto& x : X) sum += x * x;
    return sum;
}

double schwefel(const vector<double>& X){
    double sum = 0;
    for(const auto& x : X) sum += -x * sin(sqrt(fabs(x)));
    return sum;
}

double michalewicz(const vector<double>& X) {
    double sum = 0.0;
    for(int i = 0; i < X.size(); i++) {
        sum += sin(X[i]) * pow(sin((i + 1) * X[i] * X[i] / PI), 20.0);
    }
    return -sum;
}

// =========================== STRUCTURES =======================================

struct Individual {
    vector<double> x;
    double fitness;
    bool evaluated = false;

    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

// ============================ GRAY CODING ==============================

uint32_t binaryToGray(uint32_t num) { return num ^ (num >> 1); }
uint32_t grayToBinary(uint32_t gray) {
    uint32_t mask = gray;
    while (mask) { mask >>= 1; gray ^= mask; }
    return gray;
}

// =========================== ADAPTIVE OPERATOR MANAGER =======================

enum CrossoverType { ARITHMETIC, UNIFORM };
enum MutationType { GAUSSIAN, GRAY_CODE, RANDOM_RESET };

struct AdaptiveManager {
    vector<double> crossWeights = {0.5, 0.5}; 
    vector<double> mutWeights = {0.33, 0.33, 0.34}; 
    const double MIN_PROB = 0.05; 

    void update(vector<double>& weights, int idx, double reward) {
        double learningRate = 0.1;
        weights[idx] = (1.0 - learningRate) * weights[idx] + learningRate * reward;
        double sum = 0; for(double w : weights) sum += w;
        for(double &w : weights) w = max(MIN_PROB, w / sum); 
        sum = 0; for(double w : weights) sum += w;
        for(double &w : weights) w /= sum; 
    }

    int select(const vector<double>& weights) {
        double r = dist01(rng);
        double sum = 0;
        for(int i=0; i<weights.size(); ++i) {
            sum += weights[i];
            if(r <= sum) return i;
        }
        return weights.size() - 1;
    }
    
    void reset() {
        fill(crossWeights.begin(), crossWeights.end(), 0.5);
        fill(mutWeights.begin(), mutWeights.end(), 0.33);
    }
};

// =========================== GENETIC OPERATORS ==============================

// Crossover
void op_arithmetic(Individual& p1, Individual& p2, double minX, double maxX) {
    for(size_t i=0; i<p1.x.size(); ++i) {
        double avg = (p1.x[i] + p2.x[i]) / 2.0;
        p1.x[i] = avg; p2.x[i] = avg; 
    }
    p1.evaluated = p2.evaluated = false;
}

void op_uniform(Individual& p1, Individual& p2, double minX, double maxX) {
    for(size_t i=0; i<p1.x.size(); ++i) {
        if(dist01(rng) < 0.5) swap(p1.x[i], p2.x[i]);
    }
    p1.evaluated = p2.evaluated = false;
}

// Mutation
void op_gauss_mut(Individual& ind, double minX, double maxX, double pm) {
    double sigma = (maxX - minX) * 0.1;
    normal_distribution<double> d(0, sigma);
    for(int i=0; i<ind.x.size(); ++i) {
        if(dist01(rng) < pm) {
            ind.x[i] += d(rng);
            ind.x[i] = max(minX, min(maxX, ind.x[i]));
            ind.evaluated = false;
        }
    }
}

void op_gray_mut(Individual& ind, double minX, double maxX, double pm) {
    double range = maxX - minX;
    for (int i = 0; i < ind.x.size(); ++i) {
        if (dist01(rng) < pm) {
            double norm = (ind.x[i] - minX) / range;
            norm = max(0.0, min(1.0, norm));
            uint32_t discrete = static_cast<uint32_t>(norm * GRAY_PRECISION);
            uint32_t gray = binaryToGray(discrete);
            gray ^= (1U << (rand() % 32)); 
            uint32_t binary = grayToBinary(gray);
            ind.x[i] = minX + (static_cast<double>(binary) / GRAY_PRECISION) * range;
            ind.evaluated = false;
        }
    }
}

void op_random_reset(Individual& ind, double minX, double maxX, double pm) {
    for(int i=0; i<ind.x.size(); ++i) {
        if(dist01(rng) < pm) {
            ind.x[i] = minX + dist01(rng) * (maxX - minX);
            ind.evaluated = false;
        }
    }
}

// =========================== META-ALGORITHMS ==================================

Individual tournament_selection(const vector<Individual>& pop, int k) {
    int bestIdx = rand() % pop.size();
    for (int i = 1; i < k; ++i) {
        int idx = rand() % pop.size();
        if (pop[idx].fitness < pop[bestIdx].fitness) bestIdx = idx;
    }
    return pop[bestIdx];
}

void local_search(Individual& ind, function<double(const vector<double>&)> f, double minX, double maxX) {
    double step = (maxX - minX) * 0.005; 
    bool improved = true;
    int max_evals = 40; 
    int evals = 0;

    while(improved && evals < max_evals) {
        improved = false;
        for(int i=0; i<ind.x.size(); ++i) {
            double original = ind.x[i];
            
            ind.x[i] = min(maxX, original + step);
            double f_plus = f(ind.x);
            evals++;
            if(f_plus < ind.fitness) { ind.fitness = f_plus; improved = true; continue; }

            ind.x[i] = max(minX, original - step);
            double f_minus = f(ind.x);
            evals++;
            if(f_minus < ind.fitness) { ind.fitness = f_minus; improved = true; continue; }

            ind.x[i] = original; 
        }
        if(!improved && step > 1e-6) { step *= 0.5; improved = true; }
    }
    ind.evaluated = true;
}

// =========================== MAIN GA LOGIC ===================================

struct GAResult {
    double bestFitness;
    double time;
};

double calculate_diversity(const vector<Individual>& pop, int dim) {
    vector<double> center(dim, 0.0);
    for(const auto& ind : pop) for(int i=0; i<dim; ++i) center[i] += ind.x[i];
    for(int i=0; i<dim; ++i) center[i] /= pop.size();

    double div = 0;
    for(const auto& ind : pop) {
        double d = 0;
        for(int i=0; i<dim; ++i) d += pow(ind.x[i] - center[i], 2);
        div += sqrt(d);
    }
    return div / pop.size();
}

GAResult runAdaptiveGA(int dim, function<double(const vector<double>&)> func, double minX, double maxX) {
    auto start = chrono::high_resolution_clock::now();

    vector<Individual> pop(POP_SIZE);
    AdaptiveManager opManager;

    //initialization
    for(int i=0; i<POP_SIZE; ++i) {
        pop[i].x.resize(dim);
        for(int j=0; j<dim; ++j) pop[i].x[j] = minX + dist01(rng)*(maxX - minX);
        pop[i].fitness = func(pop[i].x);
        pop[i].evaluated = true;
    }

    Individual globalBest = pop[0]; 
    globalBest.fitness = 1e100;

    int stagnation = 0;
    double currentMutationRate = 1.0 / dim;

    for(int gen=0; gen<MAX_GENERATIONS; ++gen) {
        
        //sorting the population necessary for elitism
        sort(pop.begin(), pop.end()); 
        
        // update global best and stagnation counter
        if(pop[0].fitness < globalBest.fitness) {
            globalBest = pop[0];
            stagnation = 0;
        } else {
            stagnation++;
        }

        // addaptive parameters
        double diversity = calculate_diversity(pop, dim);
        if(diversity < 1e-3) currentMutationRate = min(0.3, currentMutationRate * 1.1);
        else if(diversity > 1.0) currentMutationRate = max(1.0/dim, currentMutationRate * 0.9);

        vector<Individual> newPop;
        newPop.reserve(POP_SIZE);

        //elitism
        for(int i=0; i<ELITE_SIZE; ++i) {
            newPop.push_back(pop[i]);
        }

        //the rest of the population
        while(newPop.size() < POP_SIZE) {
            

            Individual p1 = tournament_selection(pop, 4); 
            Individual p2 = tournament_selection(pop, 4);
            double parentBestFit = min(p1.fitness, p2.fitness);

            Individual c1 = p1;
            Individual c2 = p2;

            // crossover
            int crossOp = opManager.select(opManager.crossWeights);
            if(crossOp == ARITHMETIC) op_arithmetic(c1, c2, minX, maxX);
            else op_uniform(c1, c2, minX, maxX);

            // mutation
            int mutOp = opManager.select(opManager.mutWeights);
            if(mutOp == GAUSSIAN) { 
                op_gauss_mut(c1, minX, maxX, currentMutationRate); 
                op_gauss_mut(c2, minX, maxX, currentMutationRate); 
            } else if (mutOp == GRAY_CODE) {
                op_gray_mut(c1, minX, maxX, currentMutationRate);
                op_gray_mut(c2, minX, maxX, currentMutationRate);
            } else { 
                op_random_reset(c1, minX, maxX, currentMutationRate);
                op_random_reset(c2, minX, maxX, currentMutationRate);
            }

            if(!c1.evaluated) { c1.fitness = func(c1.x); c1.evaluated = true; }
            if(!c2.evaluated) { c2.fitness = func(c2.x); c2.evaluated = true; }

            // reward calculation
            double bestChild = min(c1.fitness, c2.fitness);
            double reward = (bestChild < parentBestFit) ? 1.0 : 0.0;
            
            if (reward > 0) {
                opManager.update(opManager.crossWeights, crossOp, reward);
                opManager.update(opManager.mutWeights, mutOp, reward);
            }

            newPop.push_back(c1);
            if(newPop.size() < POP_SIZE) newPop.push_back(c2);
        }

        pop = newPop;

        // ================= META-ALGORITHMS =================
        
        //local search
        if (gen % 10 == 0 || stagnation > 10) {
            local_search(pop[0], func, minX, maxX);
            if(pop[0].fitness < globalBest.fitness) globalBest = pop[0];
        }
        // restart mecanism,cataclysm
        if(stagnation > 50) {
            //we keep the best individual
            for(int i=1; i<POP_SIZE; ++i) {
                 for(int j=0; j<dim; ++j) pop[i].x[j] = minX + dist01(rng)*(maxX - minX);
                 pop[i].fitness = func(pop[i].x);
                 pop[i].evaluated = true;
            }
            stagnation = 0;
            opManager.reset();
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    return {globalBest.fitness, diff.count()};
}

int main() {
    vector<pair<string, function<double(const vector<double>&)>>> funcs = {
        {"Rastrigin", rastrigin},
        {"DeJong", deJong},
        {"Schwefel", schwefel},
        {"Michalewicz", michalewicz}
    };

    struct Bounds { double min, max; };
    vector<Bounds> bounds = { {-5.12, 5.12}, {-5.12, 5.12}, {-500, 500}, {0, PI} };

    int DIM = 30;
    
    cout << "=== Adaptive GA ===\n";
    
    for(int i=0; i<funcs.size(); ++i) {
        cout << "\nRunning " << funcs[i].first << " (D=" << DIM << ")...\n";
        
        vector<double> results;
        double totalTime = 0;

        for(int r=0; r<NUM_RUNS; ++r) {
            GAResult res = runAdaptiveGA(DIM, funcs[i].second, bounds[i].min, bounds[i].max);
            results.push_back(res.bestFitness);
            totalTime += res.time;
            
            if((r+1) % 5 == 0) cout << " Run " << r+1 << " complete.\n";
        }

        double best = *min_element(results.begin(), results.end());
        double avg = 0; for(double v : results) avg += v; avg /= NUM_RUNS;
        
        cout << "Results for " << funcs[i].first << ":\n";
        cout << " Best: " << best << "\n";
        cout << " Avg:  " << avg << "\n";
        cout << " Avg Time: " << totalTime/NUM_RUNS << "s\n";
    }

    cout << "\nDone.\n";

    return 0;
}