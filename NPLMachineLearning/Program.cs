using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using ExcelDataReader;
using IronXL;
using Microsoft.ML;
using OfficeOpenXml;
using static Microsoft.ML.DataOperationsCatalog;

namespace NPLMachineLearning
{
    class Program
    {
        static string filePath = "TrainingData.xlsx";

        static void Main(string[] args)
        {
            PredictionEngine<TrainingData, ResultData> predictionEngine = null;

            ExcelPackage.LicenseContext = LicenseContext.Commercial;
            var package = new ExcelPackage();

            predictionEngine = TrainYourModel();

            string temp = "";

            while (temp != "Kapat")
            {
                Console.WriteLine();
                Console.WriteLine("Chatbot: Ne yapmak istiyorsun? ");
                temp = Convert.ToString(Console.ReadLine());

                var prediction = predictionEngine.Predict(new TrainingData { Label = temp });

                string? result = prediction.Result;
                foreach (char c in result)
                {
                    Console.Write(c);
                    System.Threading.Thread.Sleep(20);
                }
                Console.WriteLine();

                WorkBook book = WorkBook.Load(filePath);
                var anyCell = book.DefaultWorkSheet.AllColumnsInRange[0].Where(x=>x.ToString() == temp?.ToString()).FirstOrDefault();

                if (anyCell == null)
                {
                    WorkSheet bookSheet = book.GetWorkSheet("Sheet1");

                    for (int i = 1; i <= bookSheet.RowCount + 1; i++)
                    {
                        if (bookSheet["A" + i].Value.ToString() == "0")
                        {
                            bookSheet["A" + i].Value = temp;
                            bookSheet["B" + i].Value = result;
                            break;
                        }
                    }

                    bookSheet.SaveAs(filePath);
                }

                Console.WriteLine("Chatbot: Memnun kaldınız mı? ( E / H )");
                string satisfaction = Console.ReadLine();

                if (satisfaction.ToLower() == "h")
                {
                    Console.WriteLine("Chatbot: Önerinizi yazınız: ...");

                    string suggestion = Console.ReadLine();

                    WorkBook workBook = WorkBook.Load(filePath);

                    for (int i = 1; i < workBook.DefaultWorkSheet.RowCount + 1; i++)
                    {
                        if (workBook.DefaultWorkSheet["A"+i].Value.ToString() == temp)
                        {
                            workBook.DefaultWorkSheet["B" + i].Value = suggestion.ToString();
                            break;
                        }
                    }

                    Console.WriteLine("Chatbot: Benzer olan tüm örnekler güncellensin mi? ( E / H )");
                    string selection = Console.ReadLine();

                    if (selection.ToLower() == "e")
                    {
                        var cellString = workBook.DefaultWorkSheet.AllColumnsInRange[1].Where(x => x.ToString() == result.ToString()).ToList();

                        foreach (var item in cellString)
                        {
                            item.Value = suggestion;
                        }
                    }

                    workBook.SaveAs(filePath);

                    predictionEngine = TrainYourModel();
                }
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

        static PredictionEngine<TrainingData,ResultData> TrainYourModel()
        {
            DataTable temperatureDataTable = LoadDataFromExcel(filePath);
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

            Console.WriteLine("Model Hazır.");
            Console.WriteLine();

            return predictionEngine;
        }


    }

    public class TrainingData
    {
        public string? Label { get; set; }
        public string? Desc { get; set; }
    }

    public class ResultData
    {
        public string? Result { get; set; }
    }
}
