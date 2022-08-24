/********************************************************************************
* main.cpp: Enkel implementering av ett neuralt n�tverk f�r detektering av
*           ett 2-bitars XOR-m�nster.
*
*           Kompilera programmet med GCC-kompilatorn och skapa en k�rbar fil
*           d�pt main.exe via f�ljande kommando:
*           $ g++ main.cpp dense_layer.cpp ann.cpp -o main.exe -Wall
*
*           I Windows, k�r sedan programmet via f�ljande kommando.
*           $ main.exe
*
*           I Linux, k�r programmet via f�ljande kommando:
*           $ ./main.exe
********************************************************************************/
#include "ann.hpp"

/********************************************************************************
* main: Implementerar ett neuralt n�tverk f�r detektering av ett 2-bitars
*       XOR-m�nster enligt nedan, d�r AB utg�r insignaler och X utg�r utsignal:
* 
*       A B X
*       0 0 0
*       0 1 1
*       1 0 1
*       1 1 0
* 
*       Tr�ningsdata lagras via var sin vektor och passeras till det neurala
*       n�tverket innan tr�ning. Tr�ning genomf�rs sedan under 10 000 epoker
*       med en l�rhastighet p� 1 %. Efter tr�ning predikterar det neurala
*       n�tverket angivet m�nster med 100 % precision.
********************************************************************************/
int main(void)
{
   const std::vector<std::vector<double>> train_in = 
   { 
      { 0, 0 }, 
      { 0, 1 }, 
      { 1, 0 }, 
      { 1, 1 } 
   };

   const std::vector<std::vector<double>> train_out = 
   { 
      { 0 }, 
      { 1 }, 
      { 1 }, 
      { 0 } 
   };
   
   ann ann1(2, 3, 1);
   ann1.set_training_data(train_in, train_out);
   ann1.train(10000, 0.01);
   ann1.predict();
   return 0;
}