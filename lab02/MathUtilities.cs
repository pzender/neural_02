using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace lab02
{
    public class MathUtilities
    {
        public static double Softmax(double value, List<double> vector)
        {
            return 1 / (vector.Select(v => Math.Exp(value - v)).Sum());
        }

        public static double Sigmoid(double value, List<double> vector = null)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        public static Matrix<double> DSigmoid(Matrix<double> arg)
        {
            return arg.PointwiseMultiply(1 - arg);
        }

        public static double ReLU(double value, List<double> vector = null)
        {
            return value >= 0 ? value : 0;
        }

        public static Matrix<double> DReLU(Matrix<double> arg)
        {
            return arg.Map(a => a >= 0 ? 1.0 : 0.0);
        }


        public struct ActivationFunction
        {
            public Func<double, List<double>, double> Function;
            public Func<Matrix<double>, Matrix<double>> Gradient;

        }

        public static readonly ActivationFunction SOFTMAX = new ActivationFunction()
        {
            Function = Softmax,
            Gradient = DSigmoid
        };
        public static readonly ActivationFunction SIGMOID = new ActivationFunction()
        {
            Function = Sigmoid,
            Gradient = DSigmoid
        };
        public static readonly ActivationFunction RELU = new ActivationFunction()
        {
            Function = ReLU,
            Gradient = DReLU
        };

    }
}
