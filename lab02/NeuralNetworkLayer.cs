using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace lab02
{
    class NeuralNetworkLayer
    {
        private Matrix<double> W; 
        public List<double> FeedForward(List<double> inputs)
        {
            Matrix<double> I = DenseMatrix.OfColumnArrays(inputs.Append(1).ToArray());
            I = W * I;
            I.Map(w => Sigmoid(w));

            double[] o = I.ToColumnArrays()[0];
            return o.Take(o.Count() - 1).ToList();
        }

        
        public List<double> PropagateBackward(List<double> errors, List<double> vs, double learningRate)
        {
            Matrix<double> E = DenseMatrix.OfColumnArrays(errors.ToArray());
            //TODO!!
            //Matrix<double> dW = learningRate * E.PointwiseMultiply();

            
            throw new NotImplementedException();
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
