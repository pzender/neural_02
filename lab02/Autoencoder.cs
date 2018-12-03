using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace lab02
{
    class Autoencoder
    {
        List<NeuralNetworkLayer> layers;
        private double learning_rate;
        private readonly double i_learning_rate;
        private double momentum_rate;
        private readonly double i_momentum_rate;
        private readonly double dropout_rate;
        private readonly bool adaptive_learning_rate;

        public int ExamplesProcessed { get; private set; }
        Random r = new Random();

        public void TrainExample(List<double> inputs)
        {
            ExamplesProcessed++;
            List<double> errors = Decode(Encode(inputs, false), false)
                .Zip(inputs, (predicted, actual) => actual - predicted)
                .ToList();
            for (int l = 0; l < layers.Count; l++)
            {
                errors = layers[layers.Count - l - 1].PropagateBackward(errors, learning_rate, dropout_rate);
            }

        }

        public void BatchTrain(List<List<double>> trainingData, int batchSize)
        {
            int training_data_size = 50000;
            List<List<double>> validationData = trainingData
                .Skip(training_data_size)
                .Take(1000)
                .ToList();
            double accuracy = -100000, prev_accuracy = -100000;
            for (int i = 0; this.ExamplesProcessed < 100000 && accuracy >= prev_accuracy; i++)
            {
                if (i % (1000 / batchSize) == 0)
                {
                    prev_accuracy = accuracy;
                    //Console.WriteLine("\ttesting batch on validation data");
                    accuracy = BatchTest(validationData);
                    Console.WriteLine($"{ExamplesProcessed}\t{accuracy * 100}%");
                    if (adaptive_learning_rate)
                    {
                        learning_rate = i_learning_rate * 0.1 / accuracy;
                        momentum_rate = i_momentum_rate * 0.1 / accuracy;

                    }
                }
                List<List<double>> trainingBatch = trainingData
                    .Skip((batchSize * i) % training_data_size)
                    .Take(batchSize)
                    .ToList();



                foreach (var c in trainingBatch)
                {
                    TrainExample(c);
                }
                foreach (NeuralNetworkLayer l in layers)
                {
                    l.UpdateWeights(learning_rate, momentum_rate);
                }

            }
            Console.WriteLine("Done!");
        }

        public List<double> Encode(List<double> inputs, bool useDropout)
        {
            for (int i = 0; i < layers.Count / 2; i++)
            {
                inputs = layers[i].FeedForward(inputs, useDropout);
            }
            return inputs;
        }

        public List<double> Decode(List<double> inputs, bool useDropout)
        {
            for (int i = layers.Count / 2; i < layers.Count; i++)
            {
                inputs = layers[i].FeedForward(inputs, useDropout);
            }
            return inputs;

        }

        public double BatchTest(List<List<double>> testData)
        {
            double totalError = 0;
            foreach (List<double> test_case in testData)
            {
                totalError += Decode(Encode(test_case, false), false)
                    .Zip(test_case, (predicted, actual) => actual - predicted)
                    .Select(e => e*e)
                    .Sum();
            }

            return 1 - (totalError / testData.Select(tc => tc.Sum()).Sum());


            throw new NotImplementedException();
        }


        public Autoencoder(List<int> layers, double weights_range = 0.2, double learning_rate = 0.02, double momentum_rate = 0, bool adaptive_learning_rate = false, double dropout_rate = 0)
        {
            this.layers = new List<NeuralNetworkLayer>();
            for (int i = 0; i < layers.Count - 1; i++)
            {
                this.layers.Add(new NeuralNetworkLayer(layers[i], layers[i + 1], weights_range, dropout_rate));
            }
            for (int j = layers.Count - 1; j > 0; j--)
            {
                this.layers.Add(new NeuralNetworkLayer(layers[j], layers[j - 1], weights_range, dropout_rate));
            }
            this.learning_rate = i_learning_rate = learning_rate;
            this.momentum_rate = i_momentum_rate = momentum_rate;
            this.adaptive_learning_rate = adaptive_learning_rate;
            this.dropout_rate = dropout_rate;
            ExamplesProcessed = 0;
        }

        public List<NeuralNetworkLayer> GetEncoder()
        {
            return layers.Take(layers.Count / 2).ToList();
        }

    }
}
