
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NPLMachineLearning
{

    

    class Program
    {
        private const string DATASET_PATH = "sensors_data.csv";

        private static readonly MLContext mlContext = new MLContext(2020);

        static void Main(string[] args)
        {
            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>(DATASET_PATH, separatorChar: '\t', hasHeader: true);

            var shuffledData = mlContext.Data.ShuffleRows(data, seed: 2020);

            var split = mlContext.Data.TrainTestSplit(shuffledData,testFraction: 0.2);

            var trainingData = split.TrainSet;
            var testingData = split.TestSet;

            var features = mlContext.Data.CreateEnumerable<ModelInput>(trainingData, true);


            //display(features.Take(10));
        }

    }
}

public class ModelInput
{
    [ColumnName("Temperature"), LoadColumn(0)]
    public float Temperature { get; set; }

    [ColumnName("Luminosity"), LoadColumn(1)]
    public float Luminosity { get; set; }

    [ColumnName("Infrared"), LoadColumn(2)]
    public float Infrared { get; set; }

    [ColumnName("Distance"), LoadColumn(3)]
    public float Distance { get; set; }

    [ColumnName("CreatedAt"), LoadColumn(4)]
    public string CreatedAt { get; set; }

    [ColumnName("Label"), LoadColumn(5)]
    public string Source { get; set; }
}
