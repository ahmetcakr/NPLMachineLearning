using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ChatbotExample
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<ChatData>("chat-data.csv", separatorChar: ',');

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Input", outputColumnName: "InputFeatures"))
                .Append(mlContext.Transforms.NormalizeMinMax(inputColumnName: "InputFeatures", outputColumnName: "InputFeatures"))
                .Append(mlContext.Transforms.Concatenate("InputLastFeatures", "InputFeatures"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))
                .Append(mlContext.Transforms.Text.FeaturizeText("OutputFeatures","Output"))
                .Append(mlContext.Transforms.Concatenate("Features", "InputLastFeatures", "OutputFeatures"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel","Label"));

            var model = pipeline.Fit(dataView);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ChatData, ChatPrediction>(model);

            while (true)
            {
                Console.WriteLine("Chatbot: Merhaba! Nasıl yardımcı olabilirim?");
                string input = Console.ReadLine();



                //ChatData chatData = new ChatData
                //{
                //    Input = input
                //};

                ChatPrediction prediction = predictionEngine.Predict(new ChatData { Label = input});
                

                if (prediction.Label == "Positive")
                {
                    Console.WriteLine("Chatbot: Memnun kaldınız mı? (Evet/Hayır)");
                    string satisfaction = Console.ReadLine();

                    if (satisfaction.ToLower() == "hayır")
                    {
                        Console.WriteLine("Chatbot: Önerimiz şudur: ...");
                        string suggestion = Console.ReadLine();

                        // Öneriyi veri setine ekleyin ve modeli yeniden eğitin
                    }
                }
                else
                {
                    Console.WriteLine("Chatbot: Seni anlamadım. Tekrar söyler misiniz?");
                }
            }
        }
    }

    public class ChatData
    {
        [LoadColumn(0)]
        public string Input;

        [LoadColumn(1)]
        public string Output;

        [LoadColumn(2)]
        public string Label;
    }

    public class ChatPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label;
    }
}
