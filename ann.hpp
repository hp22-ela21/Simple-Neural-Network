/********************************************************************************
* ann.hpp: Innehåller funktionalitet för att enkelt kunna realisera enklare
*          neurala nätverk via strukten ann (ANN = Artificial Neural Network).
********************************************************************************/
#ifndef ANN_HPP_
#define ANN_HPP_

/* Inkluderingsdirektiv: */
#include "dense_layer.hpp"

/********************************************************************************
* ann: Strukt för enklare neurala nätverk innehållande ett ingångslager, ett
*      dolt lager samt ett utgångslager med valfritt antal noder i respektive
*      lager. Träningsdata kan passeras via referenser till tvådimensionella
*      vektorer, vars innehåll lagras för träning.
********************************************************************************/
struct ann
{
   /* Medlemmar: */
   dense_layer hidden_layer;                         /* Dolt lager. */
   dense_layer output_layer;                         /* Utgångslager. */
   std::vector<std::vector<double>> train_in;        /* Indata för träningsuppsättningar. */
   std::vector<std::vector<double>> train_out;       /* Utdata för träningsuppsättningar. */
   std::vector<std::size_t> train_order;             /* Ordningsföljd för träningsuppsättningar. */
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