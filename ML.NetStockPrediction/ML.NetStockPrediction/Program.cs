using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Threading.Tasks;

namespace ML.NetStockPrediction
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                StockPredictionHelper stockPredictor=new StockPredictionHelper();
                stockPredictor.SaveModel("data/countries.stats.csv");
                stockPredictor.TestPrediction();
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.Write(ex.Message);
            }
        }
        
    }
}
