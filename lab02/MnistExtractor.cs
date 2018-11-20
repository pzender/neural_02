using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace lab02
{
    class MnistExtractor
    {
        private const double MAX_VALUE = 255.0;
        public static Dictionary<List<double>, int> Extract(string filename)
        {
            string line;
            Dictionary<List<double>, int> result = new Dictionary<List<double>, int>();
            StreamReader file = new StreamReader(Path.Combine(FindAppRootDir(), filename));
            while ((line = file.ReadLine()) != null)
            {
                string[] line_split = line.Split(',');
                //LABEL
                int label = int.Parse(line_split[0]);
                //IMAGE
                List<double> img = new List<double>();
                for (int i = 0; i < 28*28; i++)
                {
                    img.Add(double.Parse(line_split[i + 1]) / MAX_VALUE);   
                }
                result.Add(img, label);
            }

            
            return result;
        }


        private static string FindAppRootDir()
        {
            string path = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);

            if (path.Contains("lab02"))
            {
                return Path.Combine(path.Substring(0, path.LastIndexOf("lab02")), "lab02");
            }

            else return path;
        }

    }
}
