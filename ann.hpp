/********************************************************************************
* ann.hpp: Inneh�ller funktionalitet f�r att enkelt kunna realisera enklare
*          neurala n�tverk via strukten ann (ANN = Artificial Neural Network).
********************************************************************************/
#ifndef ANN_HPP_
#define ANN_HPP_

/* Inkluderingsdirektiv: */
#include "dense_layer.hpp"

/********************************************************************************
* ann: Strukt f�r enklare neurala n�tverk inneh�llande ett ing�ngslager, ett
*      dolt lager samt ett utg�ngslager med valfritt antal noder i respektive
*      lager. Tr�ningsdata kan passeras via referenser till tv�dimensionella
*      vektorer, vars inneh�ll lagras f�r tr�ning.
********************************************************************************/
struct ann
{
   /* Medlemmar: */
   dense_layer hidden_layer;                         /* Dolt lager. */
   dense_layer output_layer;                         /* Utg�ngslager. */
   std::vector<std::vector<double>> train_in;        /* Indata f�r tr�ningsupps�ttningar. */
   std::vector<std::vector<double>> train_out;       /* Utdata f�r tr�ningsupps�ttningar. */
   std::vector<std::size_t> train_order;             /* Ordningsf�ljd f�r tr�ningsupps�ttningar. */
   const std::vector<double>* input_layer = nullptr; /* Pekare till aktuell indata. */ 

   /* Medlemsfunktioner: */
   ann(void) { }
   ann(const std::size_t num_inputs,
       const std::size_t num_hidden,
       const std::size_t num_outputs);

   std::size_t num_inputs(void) const { return this->input_layer->size(); }
   std::size_t num_hidden(void) const { return this->hidden_layer.num_nodes(); }
   std::size_t num_outputs(void) const { return this->output_layer.num_nodes(); }
   std::size_t num_sets(void) const { return this->train_order.size(); }
   std::vector<double>& output(void) { return this->output_layer.output; }

   void set_training_data(const std::vector<std::vector<double>>& train_in,
                          const std::vector<std::vector<double>>& train_out);
   void train(const std::size_t num_epochs,
              const double learning_rate);
   std::vector<double>& predict(const std::vector<double>& input);
   void predict(std::ostream& ostream = std::cout,
                const double threshold = 0.001);
   void predict(const std::vector<std::vector<double>>& input, 
                std::ostream& ostream = std::cout,
                const double threshold = 0.001);

private:
   void shuffle(void);
   void optimize(const std::vector<double>& input, 
                 const std::vector<double>& reference,
                 const double learning_rate);
};

#endif /* ANN_HPP_ */