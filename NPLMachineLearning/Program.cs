using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Microsoft.ML;
using OfficeOpenXml;
using static Microsoft.ML.DataOperationsCatalog;

namespace NPLMachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            ExcelPackage.LicenseContext = LicenseContext.Commercial;
            var package = new ExcelPackage();

            // TrainingData must not be null
            DataTable temperatureDataTable = LoadDataFromExcel("TrainingData.xlsx");

            var temperatureData = temperatureDataTable.AsEnumerable()
                .Select(row => new TrainingData
                {
                    Label = row.Field<string>("Label"),
                    Desc = row.Field<string>("Desc")
                }).ToList();


            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromEnumerable(temperatureData);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("DescEncoded", "Desc")
                                    .Append(mlContext.Transforms.Text.FeaturizeText("LabelFeaturized", "Label"))
                                    .Append(mlContext.Transforms.Concatenate("Features", "LabelFeaturized"))
                                    .AppendCacheCheckpoint(mlContext)
                                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("DescEncoded", "Features"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("Result", "PredictedLabel"));

            var model = pipeline.Fit(dataView);

            string modelPath = "model.zip";

            mlContext.Model.Save(model, dataView.Schema, modelPath);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<TrainingData, ResultData>(model);

            Console.WriteLine("Başarılı.");

            string temp = "a";

            while (temp != "")
            {
                Console.WriteLine("Ne yapmak istiyorsun? ");
                temp = Convert.ToString(Console.ReadLine());

                var prediction = predictionEngine.Predict(new TrainingData { Label = temp });

                Console.WriteLine($"{prediction.Result}");
            }
        }

        static DataTable LoadDataFromExcel(string filePath)
        {
            using (var package = new ExcelPackage(new FileInfo(filePath)))
            {
                ExcelWorksheet worksheet = package.Workbook.Worksheets[0];
                DataTable dataTable = new DataTable();

                dataTable.Columns.Add("Label");
                dataTable.Columns.Add("Desc");

                for (int i = 2; i <= worksheet.Dimension.End.Row; i++)
                {
                    DataRow dataRow = dataTable.NewRow();
                    dataRow["Label"] = worksheet.Cells[i, 1].Value.ToString();
                    dataRow["Desc"] = worksheet.Cells[i, 2].Value.ToString();

                    dataTable.Rows.Add(dataRow);
                }

                return dataTable;
            }

        }

    }

    public class TrainingData
    {
        public string Label { get; set; }
        public string Desc { get; set; }
    }

    public class ResultData
    {
        public string Result { get; set; }
    }
}
