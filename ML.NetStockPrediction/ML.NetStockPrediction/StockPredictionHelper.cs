using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace ML.NetStockPrediction
{
    class StockPredictionHelper
    {
        /// <summary>
        /// Build model for predicting next month country unit sales using Learning Pipelines API
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <returns></returns>
        private PredictionModel<StockPredictionData, StockPrediction> CreateCountryModelUsingPipeline(string dataPath)
        {
            Console.WriteLine("*************************************************");
            Console.WriteLine("Training country forecasting model using Pipeline");

            var pipeline = new LearningPipeline
            {   
            new TextLoader(dataPath).CreateFrom<StockPredictionData>(useHeader: true, separator: ','),
            new ColumnCopier(("NetBuyVolume","Label")),
            new CategoricalOneHotVectorizer(               
                "date"
                ),
            new ColumnConcatenator(
                "Features",
                "price"
                ),
            new FastTreeRegressor()
            };


            // Finally, we train the pipeline using the training dataset set at the first stage
            var model = pipeline.Train<StockPredictionData, StockPrediction>();

            return model;
        }
        /// <summary>
        /// Train and save model for predicting next month country unit sales
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <param name="outputModelPath">Trained model path</param>
        public void SaveModel(string dataPath, string outputModelPath = "country_month_fastTreeTweedie.zip")
        {
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            var model = CreateCountryModelUsingPipeline(dataPath);

            model.WriteAsync(outputModelPath);
        }
        /// <summary>
        /// Predict samples using saved model
        /// </summary>
        /// <param name="outputModelPath">Model file path</param>
        /// <returns></returns>
        public void TestPrediction(string outputModelPath = "country_month_fastTreeTweedie.zip")
        {
            Console.WriteLine("*********************************");
            Console.WriteLine("Testing country forecasting model");

            // Read the model that has been previously saved by the method SaveModel
            var model = PredictionModel.ReadAsync<StockPredictionData, StockPrediction>(outputModelPath).Result;

            // Build sample data
            var dataSample = new StockPredictionData()
            {
                date="21/8/2018",
                //price=10.23f,
                NetBuyVolume=1000f
            };
            // Predict sample data
            var prediction = model.Predict(dataSample);
            
            Console.WriteLine($"date: {dataSample.date}, - Real value (US$): {dataSample.price}, Forecasting (US$): {prediction.price}");

            
        }


    }
}
