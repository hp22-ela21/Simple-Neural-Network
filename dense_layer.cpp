/********************************************************************************
* dense_layer.cpp: Definition av funktionsmedlemmar tillh�rande strukten 
*                  dense_layer f�r implementering av dense-lager (dolda lager
*                  samt utg�ngslager i neurala n�tverk.
********************************************************************************/
#include "dense_layer.hpp"

/********************************************************************************
* dense_layer: Initierar parametrar i ett nytt dense-lager med valbart antal
*              noder samt vikt per nod. Lagrets parametrar tilldelas l�mpliga 
*              startv�rden, d�r bias och vikter tilldelas randomiserade flyttal 
*              mellan 0 - 1, medan utdata samt avvikelser/fel f�r varje nod 
*              s�tts till 0.
* 
*              - num_nodes  : Antalet noder i det nya dense-lagret.
*              - num_weights: Antalet vikter per nod i det nya dense-lagret.
********************************************************************************/
dense_layer::dense_layer(const std::size_t num_nodes,
                         const std::size_t num_weights)
{
   this->resize(num_nodes, num_weights);
   return;
}

/********************************************************************************
* resize: Allokerar minne samt tilldelar startv�rden f�r angivet antal noder
*         samt angivet antal vikter per nod i angivet dense-lager. Eventuellt
*         tidigare inneh�ll raderas innan omallokering genomf�rs.
* 
*         - num_nodes  : Nytt antal noder i dense-lagret efter omallokering.
*         - num_weights: Nytt antal vikter per nod i dense-lagret efter 
*                        omallokering.
********************************************************************************/
void dense_layer::resize(const std::size_t num_nodes,
                         const std::size_t num_weights)
{
   this->clear();
   this->output.resize(num_nodes, 0.0);
   this->error.resize(num_nodes, 0.0);
   this->bias.resize(num_nodes, 0.0);
   this->weights.resize(num_nodes, std::vector<double>(num_weights, 0.0));

   for (std::size_t i = 0; i < this->num_nodes(); ++i)
   {
      this->bias[i] = this->get_random();

      for (std::size_t j = 0; j < this->num_weights(); ++j)
      {
         this->weights[i][j] = this->get_random();
      }
   }

   return;
}

/********************************************************************************
* clear: Nollst�ller angivet dense-lager, vilket inneb�r att lagret sedan
*        inte inneh�ller n�gra noder. Efter nollst�llning kan dense-lagrets 
*        storlek (antalet noder samt vikter per nod) st�llas in via anrop
*        av medlemsfunktionen resize.
* 
********************************************************************************/
void dense_layer::clear(void)
{
   this->output.clear();
   this->error.clear();
   this->bias.clear();
   this->weights.clear();
   return;
}

/********************************************************************************
* feedforward: Uppdaterar utdata f�r samtliga noder i angivet dense-lager
*              via angiven indata.
* 
*              - input: Ny indata som anv�nds f�r att uppdatera nodernas
*                       utdata.
********************************************************************************/
void dense_layer::feedforward(const std::vector<double>& input)
{
   for (std::size_t i = 0; i < this->num_nodes(); ++i)
   {
      double sum = this->bias[i];

      for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
      {
         sum += input[j] * this->weights[i][j];
      }

      this->output[i] = this->relu(sum);
   }

   return;
}

/********************************************************************************
* backpropagate: Ber�knar avvikelser i angivet utg�ngslager via j�mf�relse av
*                predikterad utdata samt angiven referensdata.
* 
*                - reference: Referensdata fr�n tr�ningsupps�ttningarna,
*                             vilket j�mf�rs mot predikterad utdata.
********************************************************************************/
void dense_layer::backpropagate(const std::vector<double>& reference)
{
   for (std::size_t i = 0; i < this->num_nodes() && i < reference.size(); ++i)
   {
      const auto deviation = reference[i] - this->output[i];
      this->error[i] = deviation * this->delta_relu(this->output[i]);
   }

   return;
}

/********************************************************************************
* backpropagate: Ber�knar avvikelser i angivet dolt lagret via uppm�tta 
*                avvikelser samt vikter i efterf�ljande utg�ngslager.
* 
*                - next_layer: N�sta lager i ett givet neuralt n�tverk, vilket
*                              f�r denna implementering b�r utg�ras av ett
*                              utg�ngslager.
********************************************************************************/
void dense_layer::backpropagate(const dense_layer& next_layer)
{
   for (std::size_t i = 0; i < this->num_nodes(); ++i)
   {
      auto deviation = 0.0;

      for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
      {
         deviation += next_layer.error[j] * next_layer.weights[j][i];
      }

      this->error[i] = deviation * this->delta_relu(this->output[i]);
   }

   return;
}

/********************************************************************************
* optimize: Justerar parametrar (bias och vikter) i angivet dense-lager med
*           angiven l�rhastighet via utdata fr�n f�reg�ende lager i ett givet 
*           neuralt n�tverk. Optimeringen genomf�rs i syfte att minska aktuella
*           avvikelser, vilket medf�r f�rb�ttrad prediktion.
* 
*           - input        : Utdata fr�n f�reg�ende lager.
*           - learning_rate: L�rhastigheten, som avg�r hur mycket dense-lagrets
*                            parametrar skall justeras vid uppm�tt avvikelse.
********************************************************************************/
void dense_layer::optimize(const std::vector<double>& input,
                           const double learning_rate)
{
   for (std::size_t i = 0; i < this->num_nodes(); ++i)
   {
      const auto change_rate = this->error[i] * learning_rate;
      this->bias[i] += change_rate;

      for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
      {
         this->weights[i][j] += change_rate * input[j];
      }
   }

   return;
}

/********************************************************************************
* print: Skriver ut information om angivet dense-lager i form av antalet noder,
*        antalet vikter per nod sam atuella parametrar. Utskrift sker via 
*        angiven utstr�m, d�r standardutenheten std::cout anv�nds som default 
*        f�r utskrift i terminalen. V�rden mycket n�ra noll (inom angivet
*        intervall [-threshold, threshold] avrundas till noll f�r att undvika
*        utskrift med en stor m�ngd decimaler runt nollstrecket.
* 
*        - ostream  : Angiven utstr�m (default = std::cout).
*        - threshold: Tr�skelv�rde n�ra noll, d�r samtliga predikterade v�rden
*                     inom intervallet [-threshold, threshold] avrundas till
*                     noll (default = 0.001).
********************************************************************************/
void dense_layer::print(std::ostream& ostream,
                        const double threshold)
{
   ostream << "--------------------------------------------------------------------------------\n";

   ostream << "Number of nodes: " << this->num_nodes() << "\n";
   ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

   ostream << "Output: ";
   print_parameters(this->output, ostream, threshold);

   ostream << "Error: ";
   print_parameters(this->error, ostream, threshold);

   ostream << "Bias: ";
   print_parameters(this->bias, ostream, threshold);

   ostream << "\nWeights:\n";
   for (std::size_t i = 0; i < this->num_nodes(); ++i)
   {
      ostream << "Node " << i + 1 << ": ";
      print_parameters(this->weights[i], ostream, threshold);
   }

   ostream << "--------------------------------------------------------------------------------\n\n";
   return;
}

/********************************************************************************
* print_parameters: Skriver ut parametrar lagrade i angiven vektor p� en enda
*                   rad via angiven utstr�m, d�r standardutenheten std::cout 
*                   anv�nds som default f�r utskrift i terminalen. V�rden mycket 
*                   n�ra noll (inom angivet intervall [-threshold, threshold] 
*                   avrundas till noll f�r att undvika utskrift med en stor 
*                   m�ngd decimaler runt nollstrecket.
* 
*                   - data     : Vektor inneh�llande de parametrar som skall
*                                skrivas ut.
*                   - ostream  : Angiven utstr�m (default = std::cout).
*                   - threshold: Tr�skelv�rde n�ra noll, d�r samtliga 
*                                predikterade v�rden inom intervallet 
*                                [-threshold, threshold] avrundas till noll
*                                (default = 0.001).
********************************************************************************/
void dense_layer::print_parameters(const std::vector<double>& data,
                                   std::ostream& ostream, 
                                   const double threshold)
{
   for (auto& i : data)
   {
      if (i < threshold && i > -threshold)
      {
         ostream << "0 ";
      }
      else
      {
         ostream << i << " ";
      }
   }

   ostream << "\n";
   return;
}