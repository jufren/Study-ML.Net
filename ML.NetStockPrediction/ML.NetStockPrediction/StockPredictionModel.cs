using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using ML.NetStockPrediction;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NetStockPrediction
{
    public class StockPredictionData
    {
        public StockPredictionData()
        {
        }
            // next,country,year,month,max,min,std,count,sales,med,prev
            public StockPredictionData(string date, float price, float NetBuyVolume)
        {
            this.date = date;
            this.price = price;
            this.NetBuyVolume = NetBuyVolume;
        }
        [Column("0")]
        public string date;

        
        [Column("1")]
        public float price;

        

        [Column("2")]
        public float NetBuyVolume;
    }
    // IrisPrediction is the result returned from prediction operations
    public class StockPrediction
    {
        [ColumnName("price")]
        public float price;
    }
    public class StockPredictionPredictor : IStockPredictionPredictor
    {
        /// <summary>
        /// This method demonstrates how to run prediction on one example at a time.
        /// </summary>
        public StockPrediction Predict(string modelPath, string date, float price, float NetBuyVolume)
        {
            // Load model
            var predictionEngine = CreatePredictionEngineAsync(modelPath);

            // Build country sample
            var countrySample = new StockPredictionData(date, price, NetBuyVolume);

            // Returns prediction
            return predictionEngine.Predict(countrySample);
        }

        /// <summary>
        /// This function creates a prediction engine from the model located in the <paramref name="modelPath"/>.
        /// </summary>
        private PredictionModel<StockPredictionData, StockPrediction> CreatePredictionEngineAsync(string modelPath)
        {
            PredictionModel<StockPredictionData, StockPrediction> model = PredictionModel.ReadAsync< StockPredictionData, StockPrediction>(modelPath).Result;
            return model;
        }
    }
    public interface IStockPredictionPredictor
    {
        StockPrediction Predict(string modelPath, string date, float price, float NetBuyVolume);
    }

}

