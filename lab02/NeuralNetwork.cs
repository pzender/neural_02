using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace lab02
{
    class NeuralNetwork
    {
        public NeuralNetwork(List<int> nodesByLayer)
        {
            weights = new List<Matrix<double>>(nodesByLayer.Count);
            biases = new List<Matrix<double>>(nodesByLayer.Count);

            for(int i = 0; i < nodesByLayer.Count-1; i++)
            {
                weights.Add(DenseMatrix.CreateRandom(nodesByLayer[i + 1], nodesByLayer[i], new ContinuousUniform(-0.5, 0.5)));
                biases.Add(DenseMatrix.CreateRandom(nodesByLayer[i + 1],1, new ContinuousUniform(-0.5, 0.5)));
            }
        }
        public double LearningRate { get; set; } = 0.1;
        private List<Matrix<double>> weights;
        private List<Matrix<double>> biases;
        

        public List<double> Predict(List<double> input)
        {
            Matrix<double> working = DenseMatrix.OfColumnArrays(input.ToArray());
            for(int i = 0; i < weights.Count; i++)
            {
                working = weights[i] * working;
                working = working + biases[i];
                working.Map(w => Sigmoid(w));
            }

            return working.ToColumnArrays()[0].ToList();
            //throw new NotImplementedException();
        }

        public void Train(List<double> input, List<double> expectedOutput)
        {
            List<List<double>> outputsByLayer = new List<List<double>>();

            for(int i = 0; i < weights.Count(); i++)
            {
                Matrix<double> working = DenseMatrix.OfColumnArrays(input.ToArray());
                working = weights[i] * working;
                working = working + biases[i];
                working.Map(w => Sigmoid(w));

                outputsByLayer.Add(working.ToColumnArrays()[0].ToList());
            }

            List<double> errors = Error(outputsByLayer.Last(), expectedOutput);


            for (int l = weights.Count-1; l > 1; l--)
            {
                Matrix<double> delta_weights = LearningRate
                    * DenseVector.OfEnumerable(errors).ToColumnMatrix()
                    .PointwiseMultiply(DenseVector.OfEnumerable(outputsByLayer[l]).ToColumnMatrix())
                    .PointwiseMultiply(1 - DenseVector.OfEnumerable(outputsByLayer[l]).ToColumnMatrix())
                    .Multiply(DenseVector.OfEnumerable(outputsByLayer[l - 1]).ToColumnMatrix().Transpose());

                this.weights[l] = this.weights[l] + delta_weights;

                Matrix<double> delta_biases = LearningRate
                    * DenseVector.OfEnumerable(errors).ToColumnMatrix()
                    .PointwiseMultiply(DenseVector.OfEnumerable(outputsByLayer[l]).ToColumnMatrix())
                    .PointwiseMultiply(1 - DenseVector.OfEnumerable(outputsByLayer[l]).ToColumnMatrix());

                this.biases[l] = this.biases[l] + delta_biases;

                errors = (weights[l - 1] 
                    * DenseVector.OfEnumerable(errors).ToColumnMatrix())
                    .ToColumnArrays()[0].ToList();

            }

            
        }

        private List<double>Error (List<double> actual, List<double> expected)
        {
            return expected.Zip(actual, (exp, act) => exp - act).ToList();
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        private double DSigmoid(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

    }
}
