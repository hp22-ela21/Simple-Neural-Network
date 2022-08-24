/********************************************************************************
* dense_layer.cpp: Definition av funktionsmedlemmar tillhörande strukten 
*                  dense_layer för implementering av dense-lager (dolda lager
*                  samt utgångslager i neurala nätverk.
********************************************************************************/
#include "dense_layer.hpp"

/********************************************************************************
* dense_layer: Initierar parametrar i ett nytt dense-lager med valbart antal
*              noder samt vikt per nod. Lagrets parametrar tilldelas lämpliga 
*              startvärden, där bias och vikter tilldelas randomiserade flyttal 
*              mellan 0 - 1, medan utdata samt avvikelser/fel för varje nod 
*              sätts till 0.
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
* resize: Allokerar minne samt tilldelar startvärden för angivet antal noder
*         samt angivet antal vikter per nod i angivet dense-lager. Eventuellt
*         tidigare innehåll raderas innan omallokering genomförs.
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
* clear: Nollställer angivet dense-lager, vilket innebär att lagret sedan
*        inte innehåller några noder. Efter nollställning kan dense-lagrets 
*        storlek (antalet noder samt vikter per nod) ställas in via anrop
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
* feedforward: Uppdaterar utdata för samtliga noder i angivet dense-lager
*              via angiven indata.
* 
*              - input: Ny indata som används för att uppdatera nodernas
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
* backpropagate: Beräknar avvikelser i angivet utgångslager via jämförelse av
*                predikterad utdata samt angiven referensdata.
* 
*                - reference: Referensdata från träningsuppsättningarna,
*                             vilket jämförs mot predikterad utdata.
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
* backpropagate: Beräknar avvikelser i angivet dolt lagret via uppmätta 
*                avvikelser samt vikter i efterföljande utgångslager.
* 
*                - next_layer: Nästa lager i ett givet neuralt nätverk, vilket
*                              för denna implementering bör utgöras av ett
*                              utgångslager.
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
*           angiven lärhastighet via utdata från föregående lager i ett givet 
*           neuralt nätverk. Optimeringen genomförs i syfte att minska aktuella
*           avvikelser, vilket medför förbättrad prediktion.
* 
*           - input        : Utdata från föregående lager.
*           - learning_rate: Lärhastigheten, som avgör hur mycket dense-lagrets
*                            parametrar skall justeras vid uppmätt avvikelse.
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
*        angiven utström, där standardutenheten std::cout används som default 
*        för utskrift i terminalen. Värden mycket nära noll (inom angivet
*        intervall [-threshold, threshold] avrundas till noll för att undvika
*        utskrift med en stor mängd decimaler runt nollstrecket.
* 
*        - ostream  : Angiven utström (default = std::cout).
*        - threshold: Tröskelvärde nära noll, där samtliga predikterade värden
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
* print_parameters: Skriver ut parametrar lagrade i angiven vektor på en enda
*                   rad via angiven utström, där standardutenheten std::cout 
*                   används som default för utskrift i terminalen. Värden mycket 
*                   nära noll (inom angivet intervall [-threshold, threshold] 
*                   avrundas till noll för att undvika utskrift med en stor 
*                   mängd decimaler runt nollstrecket.
* 
*                   - data     : Vektor innehållande de parametrar som skall
*                                skrivas ut.
*                   - ostream  : Angiven utström (default = std::cout).
*                   - threshold: Tröskelvärde nära noll, där samtliga 
*                                predikterade värden inom intervallet 
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