/********************************************************************************
* dense_layer.hpp: Implementering av dense-lager för dolda samt utgångslager
*                  i neurala nätverk. Funktionalitet för att genomföra
*                  feedforward, backpropagation samt optimering via gradient
*                  descent är implementerat. 
********************************************************************************/
#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <iostream>
#include <vector>

/********************************************************************************
* dense_layer: Strukt för enkel implementering av dense-lager i neurala nätverk,
*              där antalet noder samt vikter per nod kan väljas utefter behov.
*              Objekt av denna strukt kan med fördel användas för dolda samt
*              utgångslager i neurala nätverk.
********************************************************************************/
struct dense_layer
{
   /* Medlemmar: */
   std::vector<double> output;               /* Nodernas utdata. */
   std::vector<double> error;                /* Nodernas avvikelser. */
   std::vector<double> bias;                 /* Nodernas vilovärden (m-värden). */
   std::vector<std::vector<double>> weights; /* Nodernas vikter (k-värden). */

   /* Medlemsfunktioner: */
   dense_layer(void) { }
   dense_layer(const std::size_t num_nodes,
               const std::size_t num_weights);
   std::size_t num_nodes(void) const { return this->output.size(); }
   std::size_t num_weights(void) const { return this->weights[0].size(); }

   void resize(const std::size_t num_nodes,
               const std::size_t num_weights);
   void clear(void);

   void feedforward(const std::vector<double>& input);
   void backpropagate(const std::vector<double>& reference);
   void backpropagate(const dense_layer& next_layer);
   void optimize(const std::vector<double>& input, const double learning_rate);

   void print(std::ostream& ostream = std::cout, 
              const double threshold = 0.001);
   static void print_parameters(const std::vector<double>& data,
                                std::ostream& ostream = std::cout,
                                const double threshold = 0.001);
private:
   static double get_random(void) { return std::rand() / static_cast<double>(RAND_MAX); }
   static double relu(const double sum) { return sum > 0.0 ? sum : 0.0; }
   static double delta_relu(const double output) { return output > 0.0 ? 1.0 : 0.0; }
};

#endif /* DENSE_LAYER_HPP_ */