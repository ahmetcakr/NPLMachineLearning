using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using OfficeOpenXml;
using static Microsoft.ML.DataOperationsCatalog;
using LicenseContext = OfficeOpenXml.LicenseContext;

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
                    Desc = row.Field<string>("Desc"),
                    Rating = row.Field<string>("Rating")
                }).ToList();


            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromEnumerable(temperatureData);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("DescEncoded", "Desc")
                                    .Append(mlContext.Transforms.Text.FeaturizeText("RatingEncoded", "Rating"))
                                    .Append(mlContext.Transforms.Text.FeaturizeText("LabelFeaturized", "Label"))
                                    .Append(mlContext.Transforms.Concatenate("Features", "LabelFeaturized","RatingEncoded"))
                                    .AppendCacheCheckpoint(mlContext)
                                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("DescEncoded", "Features"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("Result", "PredictedLabel"));

            var model = pipeline.Fit(dataView);

            string modelPath = "model.zip";

            mlContext.Model.Save(model, dataView.Schema, modelPath);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<TrainingData, ResultData>(model);

            Console.WriteLine("Başarılı.");
            Console.WriteLine();

            string temp = "a";

            while (temp != "")
            {
                Console.WriteLine();
                Console.WriteLine("Ne yapmak istiyorsun? ");
                temp = Convert.ToString(Console.ReadLine());

                var prediction = predictionEngine.Predict(new TrainingData { Label = temp });

                string result = prediction.Result;

                foreach (char c in result)
                {
                    Console.Write(c);
                    System.Threading.Thread.Sleep(20);
                }
                Console.WriteLine();
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
                dataTable.Columns.Add("Rating");

                for (int i = 3; i <= worksheet.Dimension.End.Row; i++)
                {
                    DataRow dataRow = dataTable.NewRow();
                    dataRow["Label"] = worksheet.Cells[i, 1].Value.ToString();
                    dataRow["Desc"] = worksheet.Cells[i, 2].Value.ToString();
                    dataRow["Rating"] = worksheet.Cells[i, 3].Value.ToString();

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
        public string Rating { get; set; }
    }

    public class ResultData
    {
        public string Result { get; set; }
        [ColumnName("PredictedRating")]
        public string Rate { get; set; }
    }
}
