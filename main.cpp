/********************************************************************************
* main.cpp: Enkel implementering av ett neuralt nätverk för detektering av
*           ett 2-bitars XOR-mönster.
*
*           Kompilera programmet med GCC-kompilatorn och skapa en körbar fil
*           döpt main.exe via följande kommando:
*           $ g++ main.cpp dense_layer.cpp ann.cpp -o main.exe -Wall
*
*           I Windows, kör sedan programmet via följande kommando.
*           $ main.exe
*
*           I Linux, kör programmet via följande kommando:
*           $ ./main.exe
********************************************************************************/
#include "ann.hpp"

/********************************************************************************
* main: Implementerar ett neuralt nätverk för detektering av ett 2-bitars
*       XOR-mönster enligt nedan, där AB utgör insignaler och X utgör utsignal:
* 
*       A B X
*       0 0 0
*       0 1 1
*       1 0 1
*       1 1 0
* 
*       Träningsdata lagras via var sin vektor och passeras till det neurala
*       nätverket innan träning. Träning genomförs sedan under 10 000 epoker
*       med en lärhastighet på 1 %. Efter träning predikterar det neurala
*       nätverket angivet mönster med 100 % precision.
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