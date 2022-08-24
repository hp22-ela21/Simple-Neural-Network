/********************************************************************************
* ann.cpp: Definition av funktionsmedlemmar tillh�rande strukten ann f�r
*          implementering av enkla neurala n�tverk best�ende av ett enda
*          dolt lager.
********************************************************************************/
#include "ann.hpp"

/********************************************************************************
* ann: Initierar nytt neuralt n�tverk med angivet antal noder i ing�ngslagret,
*      det dolda lagret samt utg�ngslagret.
* 
*      - num_inputs : Antalet noder i ing�ngslagret.
*      - num_hidden : Antalet noder i det dolda lagret.
*      - num_outputs: Antalet noder i utg�ngslagret.
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
* set_training_data: L�ser in tr�ningsdata f�r angivet neuralt n�tverk via
*                    passerade in- och utsignaler, tillsammans med att index
*                    f�r respektive tr�ningsupps�ttning lagras.
*
*                    - train_in : Inneh�ller indata f�r tr�ningsupps�ttningar.
*                    - train_out: Inneh�ller utdata f�r tr�ningsupps�ttningar.
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
* train: Tr�nar angivet neuralt n�tverk med befintlig tr�ningsdata under angivet
*        antal epoker med angiven l�rhastighet. I b�rjan av varje epok
*        randomiseras ordningen p� tr�ningsupps�ttningarna f�r att undvika att
*        eventuella icke avsedda m�nster i tr�ningsdatan p�verkar resultatet.
*        Under varje epok uts�tts det neurala n�tverket f�r samtliga befintliga 
*        tr�ningsupps�ttningar och justeras utefter prediktionsresultatet.      
*
*        - num_epochs   : Antalet epoker/omg�ngar som tr�ning skall genomf�ras.
*        - learning_rate: L�rhastigheten, som avg�r hur stor andel av uppm�tt
*                         avvikelse som n�tverkets parametrar justeras med.
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
* predict: Genomf�r prediktion med angivet neuralt n�tverk f�r angiven 
*          kombination av indata.
* 
*          - input: Vektor inneh�llande den kombination av indata som
*                   prediktion skall ske utefter.
********************************************************************************/
std::vector<double>& ann::predict(const std::vector<double>& input)
{
   this->hidden_layer.feedforward(input);
   this->output_layer.feedforward(this->hidden_layer.output);
   return this->output_layer.output;
}

/********************************************************************************
* predict: Genomf�r prediktion med angivet neuralt n�tverk via indata fr�n
*          samtliga befintliga tr�ningsupps�ttninsupps�ttningar och skriver
*          varje kombination av indata samt motsvarande predikterad utdata via
*          angiven utstr�m, d�r standardutenheten std::cout anv�nds som default
*          f�r utskrift i terminalen. V�rden mycket n�ra noll (inom angivet
*          tr�skelv�rde [-threshold, threshold] avrundas till noll f�r att
*          undvika utskrift med en stor m�ngd decimaler runt nollstrecket.
*
*          - ostream  : Angiven utstr�m (default = std::cout).
*          - threshold: Tr�skelv�rde n�r noll, d�r samtliga predikterade
*                       v�rden mellan [-threshold, threshold]  avrundas till
*                       noll (default = 0.001).
********************************************************************************/
void ann::predict(std::ostream& ostream,
                  const double threshold)
{
   this->predict(this->train_in);
   return;
}

/********************************************************************************
* predict: Genomf�r prediktion med angivet neuralt n�tverk via indata fr�n
*          samtliga befintliga tr�ningsupps�ttninsupps�ttningar och skriver
*          varje kombination av indata samt motsvarande predikterad utdata via
*          angiven utstr�m, d�r standardutenheten std::cout anv�nds som default
*          f�r utskrift i terminalen. V�rden mycket n�ra noll (inom angivet
*          intervall [-threshold, threshold] avrundas till noll f�r att undvika
*          utskrift med en stor m�ngd decimaler runt nollstrecket
*
*          - input    : Angivna kombinationer av indata som prediktion skall
*                       genomf�ras utefter.
*          - ostream  : Angiven utstr�m (default = std::cout).
*          - threshold: Tr�skelv�rde n�ra noll, d�r samtliga predikterade v�rden
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
* shuffle: Randomiserar den inb�rdes ordningen p� tr�ningsupps�ttningarna f�r
*          angivet neuralt n�tverk, vilket genomf�rs i syfte att minska risken 
*          f�r att eventuella icke avsedda m�nster som i tr�ningsdatan skall
*          p�verka tr�ningen.
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
* optimize: Ber�knar aktuella avvikelser f�r angivet neuralt n�tver med angiven 
*           tr�ningsdata och justerar n�tverkets parametrar d�refter.
*
*           input        : Kombination av indata fr�n tr�ningsdata som
*                          prediktion skall genomf�ras utefter.
*           reference    : Referensv�rde fr�n tr�ningsdatan, vilket utg�r det
*                          v�rde som n�tverket �nskas prediktera.
*           learning_rate: L�rhastigheten, som avg�r hur mycket n�tverkets
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
