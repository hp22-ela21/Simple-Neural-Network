/********************************************************************************
* ann.cpp: Definition av funktionsmedlemmar tillhörande strukten ann för
*          implementering av enkla neurala nätverk bestående av ett enda
*          dolt lager.
********************************************************************************/
#include "ann.hpp"

/********************************************************************************
* ann: Initierar nytt neuralt nätverk med angivet antal noder i ingångslagret,
*      det dolda lagret samt utgångslagret.
* 
*      - num_inputs : Antalet noder i ingångslagret.
*      - num_hidden : Antalet noder i det dolda lagret.
*      - num_outputs: Antalet noder i utgångslagret.
********************************************************************************/
ann::ann(const std::size_t num_inputs,
         const std::size_t num_hidden,
         const std::size_t num_outputs)
{
   this->hidden_layer.resize(num_hidden, num_inputs);
   this->output_layer.resize(num_outputs, num_hidden);
   return;
}

/********************************************************************************
* set_training_data: Läser in träningsdata för angivet neuralt nätverk via
*                    passerade in- och utsignaler, tillsammans med att index
*                    för respektive träningsuppsättning lagras.
*
*                    - train_in : Innehåller indata för träningsuppsättningar.
*                    - train_out: Innehåller utdata för träningsuppsättningar.
********************************************************************************/
void ann::set_training_data(const std::vector<std::vector<double>>& train_in,
                            const std::vector<std::vector<double>>& train_out)
{
   const auto num_sets = train_in.size() <= train_out.size() ? train_in.size() : train_out.size();

   this->train_in.resize(num_sets);
   this->train_out.resize(num_sets);
   this->train_order.resize(num_sets);

   for (std::size_t i = 0; i < num_sets; ++i)
   {
      this->train_in[i] = train_in[i];
      this->train_out[i] = train_out[i];
      this->train_order[i] = i;
   }

   return;
}

/********************************************************************************
* train: Tränar angivet neuralt nätverk med befintlig träningsdata under angivet
*        antal epoker med angiven lärhastighet. I början av varje epok
*        randomiseras ordningen på träningsuppsättningarna för att undvika att
*        eventuella icke avsedda mönster i träningsdatan påverkar resultatet.
*        Under varje epok utsätts det neurala nätverket för samtliga befintliga 
*        träningsuppsättningar och justeras utefter prediktionsresultatet.      
*
*        - num_epochs   : Antalet epoker/omgångar som träning skall genomföras.
*        - learning_rate: Lärhastigheten, som avgör hur stor andel av uppmätt
*                         avvikelse som nätverkets parametrar justeras med.
********************************************************************************/
void ann::train(const std::size_t num_epochs,
                const double learning_rate)
{
   for (std::size_t i = 0; i < num_epochs; ++i)
   {
      this->shuffle();

      for (std::size_t j = 0; j < this->num_sets(); ++j)
      {
         const auto k = this->train_order[j];
         this->optimize(this->train_in[k], this->train_out[k], learning_rate);
      }
   }

   return;
}

/********************************************************************************
* predict: Genomför prediktion med angivet neuralt nätverk för angiven 
*          kombination av indata.
* 
*          - input: Vektor innehållande den kombination av indata som
*                   prediktion skall ske utefter.
********************************************************************************/
std::vector<double>& ann::predict(const std::vector<double>& input)
{
   this->hidden_layer.feedforward(input);
   this->output_layer.feedforward(this->hidden_layer.output);
   return this->output_layer.output;
}

/********************************************************************************
* predict: Genomför prediktion med angivet neuralt nätverk via indata från
*          samtliga befintliga träningsuppsättninsuppsättningar och skriver
*          varje kombination av indata samt motsvarande predikterad utdata via
*          angiven utström, där standardutenheten std::cout används som default
*          för utskrift i terminalen. Värden mycket nära noll (inom angivet
*          tröskelvärde [-threshold, threshold] avrundas till noll för att
*          undvika utskrift med en stor mängd decimaler runt nollstrecket.
*
*          - ostream  : Angiven utström (default = std::cout).
*          - threshold: Tröskelvärde när noll, där samtliga predikterade
*                       värden mellan [-threshold, threshold]  avrundas till
*                       noll (default = 0.001).
********************************************************************************/
void ann::predict(std::ostream& ostream,
                  const double threshold)
{
   this->predict(this->train_in);
   return;
}

/********************************************************************************
* predict: Genomför prediktion med angivet neuralt nätverk via indata från
*          samtliga befintliga träningsuppsättninsuppsättningar och skriver
*          varje kombination av indata samt motsvarande predikterad utdata via
*          angiven utström, där standardutenheten std::cout används som default
*          för utskrift i terminalen. Värden mycket nära noll (inom angivet
*          intervall [-threshold, threshold] avrundas till noll för att undvika
*          utskrift med en stor mängd decimaler runt nollstrecket
*
*          - input    : Angivna kombinationer av indata som prediktion skall
*                       genomföras utefter.
*          - ostream  : Angiven utström (default = std::cout).
*          - threshold: Tröskelvärde nära noll, där samtliga predikterade värden
*                       inom intervallet [-threshold, threshold] avrundas till
*                       noll (default = 0.001).
********************************************************************************/
void ann::predict(const std::vector<std::vector<double>>& input,
                  std::ostream& ostream,
                  const double threshold)
{
   const auto* end = &input[input.size() - 1];
   ostream << "--------------------------------------------------------------------------------\n";

   for (auto& i : input)
   {
      const auto& prediction = this->predict(i);
      ostream << "Input: ";
      dense_layer::print_parameters(i, ostream, threshold);

      ostream << "Output: ";
      dense_layer::print_parameters(prediction, ostream, threshold);
      if (&i < end) ostream << "\n";
   }

   ostream << "--------------------------------------------------------------------------------\n\n";
   return;
}

/********************************************************************************
* shuffle: Randomiserar den inbördes ordningen på träningsuppsättningarna för
*          angivet neuralt nätverk, vilket genomförs i syfte att minska risken 
*          för att eventuella icke avsedda mönster som i träningsdatan skall
*          påverka träningen.
********************************************************************************/
void ann::shuffle(void)
{
   for (std::size_t i = 0; i < this->num_sets(); ++i)
   {
      const auto r = std::rand() % this->num_sets();
      const auto temp = this->train_order[i];
      this->train_order[i] = this->train_order[r];
      this->train_order[r] = temp;
   }

   return;
}

/********************************************************************************
* optimize: Beräknar aktuella avvikelser för angivet neuralt nätver med angiven 
*           träningsdata och justerar nätverkets parametrar därefter.
*
*           input        : Kombination av indata från träningsdata som
*                          prediktion skall genomföras utefter.
*           reference    : Referensvärde från träningsdatan, vilket utgör det
*                          värde som nätverket önskas prediktera.
*           learning_rate: Lärhastigheten, som avgör hur mycket nätverkets
*                          parametrar justeras vid fel.
********************************************************************************/
void ann::optimize(const std::vector<double>& input,
                   const std::vector<double>& reference,
                   const double learning_rate)
{
   this->input_layer = &input;
   this->hidden_layer.feedforward(*this->input_layer);
   this->output_layer.feedforward(hidden_layer.output);

   this->output_layer.backpropagate(reference);
   this->hidden_layer.backpropagate(output_layer);

   this->output_layer.optimize(this->hidden_layer.output, learning_rate);
   this->hidden_layer.optimize(*this->input_layer, learning_rate);
   return;
}
