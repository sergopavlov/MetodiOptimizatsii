using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MetodiOptimizatsii
{
    public enum TypesOfPenalty
    {
        Quadratic,
        Linear
    }
    public enum TypesOfBoundary
    {
        Logarythmic,
        Fractional
    }
    class Program
    {
        static void Main(string[] args)
        {
            /*          #region shit
                      double[] C = new double[] { 1, 2, 10, 5, 7, 9 };
                      double[] A = new double[] { 0, 0, 3, -7, 6, 6 };
                      double[] B = new double[] { -1, -4, -2, -6, -10, 1 };
                      Func<Vector, double> fank = (Vector x) =>
                           {
                               double res = 0;
                               for (int i = 0; i < 6; i++)
                               {
                                   res += -C[i] / (1 + (x.v[0] - A[i]) * (x.v[0] - A[i]) + (x.v[1] - B[i]) * (x.v[1] - B[i]));
                               }
                               return res;
                           };
                      //List<Restriction> rests = new();
                      //List<UnequalityRestriction> rests2 = new();
                      //rests.Add(new UnequalityRestriction((Vector x) => x.v[0] + x.v[1] + 1));
                      //rests.Add(new EqualityRestriction((Vector x) => x.v[1] - x.v[0] - 1));
                      #region quadratic
                      Func<Vector, double> ffunkk = (Vector x) => 4 * (x.v[1] - x.v[0]) * (x.v[1] - x.v[0]) + 3 * (x.v[0] - 1) * (x.v[0] - 1);
                      List<Func<Vector, double>> grad = new();
                      grad.Add((x) => -8 * (x.v[1] - x.v[0]) + 6 * (x.v[0] - 1));
                      grad.Add((x) => 8 * (x.v[1] - x.v[0]));
                      List<List<Func<Vector, double>>> Gesse = new();
                      Gesse.Add(new());
                      Gesse.Add(new());
                      Gesse[0].Add((x) => 14);
                      Gesse[0].Add((x) => -8);
                      Gesse[1].Add((x) => -8);
                      Gesse[1].Add((x) => 8);
                      function func = new function(ffunkk, 2, grad, Gesse);
                      #endregion
                      #region NorQuadratic   
                      Func<Vector, double> funcknonquad = (x) => 100 * (x.v[1] - x.v[0] * x.v[0]) * (x.v[1] - x.v[0] * x.v[0]) + (1 - x.v[0]) * (1 - x.v[0]);
                      List<Func<Vector, double>> grad2 = new();
                      grad2.Add((x) => -400 * x.v[0] * (x.v[1] - x.v[0] * x.v[0]) - 2 * (1 - x.v[0]));
                      grad2.Add((x) => 200 * (x.v[1] - x.v[0] * x.v[0]));
                      List<List<Func<Vector, double>>> Gesse2 = new();
                      Gesse2.Add(new());
                      Gesse2.Add(new());
                      Gesse2[0].Add((x) => -400 * x.v[1] + 1200 * x.v[0] * x.v[0] + 2);
                      Gesse2[0].Add((x) => -400 * x.v[0]);
                      Gesse2[1].Add((x) => -400 * x.v[0]);
                      Gesse2[1].Add((x) => 200);
                      function funcnonquad = new function(funcknonquad, 2, grad2, Gesse2);
                      #endregion
                      //function slojno = new function(fank, 2);
                      //slojno.NumericDerivatives = true;
                      //slojno.diffeps = 1e-4;
                      func.NumericDerivatives = false;
                      funcnonquad.NumericDerivatives = false;
                      Vector x0 = new Vector(2, 0);
                      x0.v[0] = 5;
                      x0.v[1] = -5;
                      func.SearchType = function.OneDimensionSearch.GoldenRatio;
                      funcnonquad.SearchType = function.OneDimensionSearch.GoldenRatio;
                      var r4 = Methods.Newton(funcnonquad, x0, 1e-7, 100);
                      Console.WriteLine($"{1e-7} {r4} {funcnonquad.counter}");
                      //var res = Methods.PenaltyFunctions(func, x0, rests, 1e-6, 10000, TypesOfPenalty.Quadratic);//-1 0
                      //var res2 = Methods.BoudaryFunctions(func, x0, rests2, 1e-15, 10000, TypesOfBoundary.Fractional);//-0.2632 -0.7368
                      //Vector a = new Vector(2, -10);
                      //Vector b = new Vector(2, 10);
                      //slojno.counter = 0;

                      //var asdasd1 = Methods.SimpleRandomSearch(slojno, a, b, 1e0, 0.2);
                      //Console.WriteLine(slojno.counter - 1);
                      //slojno.counter = 0;
                      //var r1 = slojno.Func(asdasd1);

                      //var asdasd2 = Methods.RandomSearch1(slojno, a, b, 10, 1e-2, x0);
                      //Console.WriteLine($"{asdasd2} {-slojno.Func(asdasd2)} {slojno.counter - 1}");
                      //slojno.counter = 0;
                      //var r2 = slojno.Func(asdasd2);

                      //var asdasd3 = Methods.RandomSearch3(slojno, x0, 10, 1e-2);
                      //Console.WriteLine($"{asdasd3} {-slojno.Func(asdasd3)} {slojno.counter - 1}");
                      //slojno.counter = 0;
                      //var r3 = slojno.Func(asdasd3);
                      #endregion
          */
            int n = 2;
            int m = 3;
            List<double> Q = new();
            Q.Add(1);
            Q.Add(-2);
            Q.Add(0);
            List<List<double>> x = new();
            x.Add(new());
            x[0].Add(1);
            x[0].Add(-1);
            x[0].Add(0);
            x.Add(new());
            x[1].Add(-2);
            x[1].Add(-1);
            x[1].Add(3);
            x.Add(new());
            x[2].Add(-1);
            x[2].Add(1);
            x[2].Add(1);
            Methods.SimplexMethod(n, Q, m, x);
            Console.WriteLine("Hello World!");
        }
    }
    public static class Methods
    {
        static Random rand = new Random(Guid.NewGuid().GetHashCode());
        public static double Goldenratio(Func<double, double> func, double a, double b, double eps)
        {
            double k1 = (3 - Math.Sqrt(5)) / 2;
            double k2 = (Math.Sqrt(5) - 1) / 2;
            bool flag = true;
            double x1 = 0, x2 = 0;
            int i = 1;
            int lastchosen = 0;
            int k = 0;
            while (flag)
            {
                if (i == 1)
                {
                    x1 = a + k1 * (b - a);
                    x2 = a + k2 * (b - a);
                }
                else
                {
                    if (lastchosen == 1)
                    {
                        x1 = x2;
                        x2 = a + k2 * (b - a);
                    }
                    else
                    {
                        x2 = x1;
                        x1 = a + k1 * (b - a);
                    }
                }
                double f1 = func(x1);
                double f2 = func(x2);
                if (b - a < eps)
                {
                    flag = false;
                }
                else
                {
                    if (Math.Abs((f1 - f2) / f1) < 1e-15)
                    {
                        a = x1;
                        b = x2;
                    }
                    else
                    {
                        i++;
                        if (f1 < f2)
                        {
                            b = x2;
                            lastchosen = 2;
                        }
                        else
                        {
                            a = x1;
                            lastchosen = 1;
                        }
                    }
                }
            }
            return (x1 + x2) / 2;
        }
        public static double QuadraticInterpolation(Func<double, double> func, double x0, double eps)
        {
            int maxiter = 100000;
            int k = 0;
            //Console.WriteLine($"{x0}");
            double h = eps;
            double f0 = func(x0 - h), f1 = func(x0), f2 = func(x0 + h);
            double b = (f0 * (2 * x0 + h) / (2 * h * h) - f1 * (2 * x0) / (h * h) + f2 * (2 * x0 - h));
            double a = (f0 / (h * h) - 2 * f1 / (h * h) + f2 / (h * h));
            if (a == 0)
            {
                return x0;
            }
            double xnext = (f0 * (2 * x0 + h) / (2 * h * h) - f1 * (2 * x0) / (h * h) + f2 * (2 * x0 - h) / (2 * h * h)) / (f0 / (h * h) - 2 * f1 / (h * h) + f2 / (h * h));
            //Console.WriteLine($"{xnext}");
            k++;
            while (Math.Abs(x0 - xnext) > eps && k < maxiter)
            {
                x0 = xnext;
                f0 = func(x0 - h);
                f1 = func(x0);
                f2 = func(x0 + h);
                b = (f0 * (2 * x0 + h) / (2 * h * h) - f1 * (2 * x0) / (h * h) + f2 * (2 * x0 - h) / (2 * h * h));
                a = (f0 / (h * h) - 2 * f1 / (h * h) + f2 / (h * h));
                if (a != 0)
                    xnext = (f0 * (2 * x0 + h) / (2 * h * h) - f1 * (2 * x0) / (h * h) + f2 * (2 * x0 - h) / (2 * h * h)) / (f0 / (h * h) - 2 * f1 / (h * h) + f2 / (h * h));
                else
                    xnext = x0;
                k++;
                //Console.WriteLine($"{xnext}");
            }
            Console.WriteLine($"Итерации метода парабол {k}");
            return xnext;
        }
        public static void findsectionwithminimum(Func<double, double> func, out double a, out double b, double x0, double delta, int maxiter)
        {
            a = 0;
            b = 0;
            double x1 = x0 + delta;
            double x2 = 0, f0, f1, f2;
            double h;
            f0 = func(x0);
            f1 = func(x1);
            if (f0 > f1)
            {
                h = delta;
            }
            else
            {
                h = -delta;
                x1 = x0 - delta;
                f1 = func(x1);
            }
            int k = 3;
            bool flag = true;
            while (flag && k < maxiter)
            {
                h *= 2;
                x2 = x1 + h;
                f2 = func(x2);
                k++;
                if (f2 > f1)
                {
                    a = Math.Min(x0, x2);
                    b = Math.Max(x0, x2);
                    flag = false;
                }
                else
                {
                    x0 = x1;
                    x1 = x2;
                    f1 = f2;
                }
            }
            if (k == maxiter)
            {
                a = Math.Min(x0, x2);
                b = Math.Max(x0, x2);
                if (double.IsNaN(a) || double.IsInfinity(a) || double.IsNaN(b) || double.IsInfinity(b))
                    throw new Exception();
            }
        }
        public static void SolveSlae(Matrix A, Vector b)
        {
            int n = b.dim;
            for (int i = 0; i < n; i++)
            {
                double summd = 0;
                for (int j = 0; j < i; j++)
                {
                    double summl = 0;
                    double summu = 0;
                    for (int k = 0; k < j; k++)
                    {
                        summl += A.m[i][k] * A.m[k][j];
                        summu += A.m[j][k] * A.m[k][i];
                    }
                    A.m[i][j] -= summl;
                    A.m[j][i] = (A.m[j][i] - summu) / A.m[j][j];
                    summd += A.m[i][j] * A.m[j][i];
                }
                A.m[i][i] -= summd;
            }
            for (int i = 0; i < n; i++)
            {
                double summ = 0;
                for (int j = 0; j < i; j++)
                {
                    summ += A.m[i][j] * b.v[j];
                }
                b.v[i] = (b.v[i] - summ) / A.m[i][i];
            }
            for (int i = n - 1; i >= 0; i--)
            {
                double summ = 0;
                for (int j = n - 1; j > i; j--)
                {
                    summ += A.m[i][j] * b.v[j];
                }
                b.v[i] -= summ;
            }
        }
        public static Vector Newton(function func, Vector x0, double eps, int maxiter)
        {
            int n = x0.dim;
            int k = 0;
            Matrix A = func.Gesse(x0);//1+n+n^2
            Vector b = -func.grad(x0);//n+1
            //Console.WriteLine($"Newton {k} {x0}");
            while (b.norm > eps && k < maxiter)
            {
                SolveSlae(A, b);
                x0 = x0 + b * func.DirectionMinimum(x0, b);
                k++;
                //Console.WriteLine($"{k} {x0}");
                A = func.Gesse(x0);
                b = -func.grad(x0);
            }
            Console.WriteLine($"Newton iterations: {k}");
            return x0;
        }
        public static Vector DavidonFletcherPauel(function func, Vector x0, double eps, int maxiter, double omega)
        {
            int n = x0.dim;
            Matrix Eta = new Matrix(n, 1);
            int k = 0;
            Console.WriteLine($"{k} {x0}");
            Vector lastgrad = func.grad(x0);
            Vector xlast = x0;
            Vector deltax = omega * (Eta * lastgrad);
            double lambda = func.DirectionMinimum(x0, deltax);
            Vector curx = x0 + lambda * deltax;
            deltax *= lambda;
            Vector curgrad = func.grad(curx);
            Vector deltagrad = curgrad - lastgrad;
            k++;
            Console.WriteLine($"{k} {curx}");
            while (curgrad.norm > eps && k < maxiter)
            {
                Eta += ((deltax ^ deltax) / (deltax * deltagrad)) / omega - Eta * (deltagrad ^ deltagrad) * Eta.Transpose() / (deltagrad * (Eta * deltagrad));
                xlast = curx;
                lastgrad = curgrad;
                deltax = omega * (Eta * lastgrad);
                lambda = func.DirectionMinimum(xlast, deltax);
                curx = xlast + lambda * deltax;
                deltax *= lambda;
                curgrad = func.grad(curx);
                deltagrad = curgrad - lastgrad;
                k++;
                Console.WriteLine($"{k} {curx}");
            }
            return curx;
        }
        public static Vector Broyden(function func, Vector x0, double eps, int maxiter)
        {
            int n = x0.dim;
            Matrix Eta = new Matrix(n, 1);
            int k = 0;
            Vector lastgrad = func.grad(x0);
            Vector xlast = x0;
            Vector deltax = Eta * lastgrad;
            double lambda = func.DirectionMinimum(x0, deltax);
            Vector curx = x0 + lambda * deltax;
            deltax *= lambda;
            Vector curgrad = func.grad(curx);
            Vector deltagrad = curgrad - lastgrad;
            k++;
            while (curgrad.norm > eps && k < maxiter)
            {
                Eta += ((deltax - Eta * deltagrad) ^ (deltax - Eta * deltagrad)) / ((deltax - Eta * deltagrad) * deltagrad);
                xlast = curx;
                lastgrad = curgrad;
                deltax = (Eta * lastgrad);
                lambda = func.DirectionMinimum(xlast, deltax);
                curx = xlast + lambda * deltax;
                deltax *= lambda;
                curgrad = func.grad(curx);
                deltagrad = curgrad - lastgrad;
                k++;
            }
            return curx;
        }
        public static Vector Gauss(function func, Vector x0, double eps, int maxiter)
        {
            double flast = 0;
            int k = 0;
            double fcur = func.Func(x0);
            int n = func.dim;
            Vector dir = new Vector(n, 0);
            Vector curpoint = x0;
            do
            {
                flast = fcur;
                for (int i = 0; i < n; i++)
                {
                    dir.v[i] = 1;
                    curpoint = curpoint + func.DirectionMinimum(curpoint, dir) * dir;
                    dir.v[i] = 0;
                }
                fcur = func.Func(curpoint);
                k++;
            } while (Math.Abs(fcur - flast) > eps && k < maxiter);
            return curpoint;
        }
        public static Vector PenaltyFunctions(function func, Vector x0, List<Restriction> restrictions, double eps, int maxiter, TypesOfPenalty penaltytype)
        {
            double penaltymultiplier = 1;
            Vector curpoint = x0;
            int k = 0;
            int n = x0.dim;
            double curpenalty = CalcPenalty(curpoint, restrictions, penaltytype);
            do
            {
                function pfunc = new function(func.Func, restrictions, penaltymultiplier, n, penaltytype);
                curpoint = Gauss(pfunc, curpoint, 1e-15, maxiter);
                curpenalty = CalcPenalty(curpoint, restrictions, penaltytype);
                k++;
                penaltymultiplier += k;
                Console.WriteLine($"{k} {curpoint} {curpenalty}");
            } while (curpenalty > eps && k < maxiter);
            return curpoint;
        }
        public static Vector BoudaryFunctions(function func, Vector x0, List<UnequalityRestriction> restrictions, double eps, int maxiter, TypesOfBoundary type)
        {
            double penaltymultiplier = 100;
            Vector curpoint;
            Vector lastpoint = x0;
            int k = 0;
            int n = x0.dim;
            bool flag = true;
            double curpenalty = CalcPenalty(lastpoint, restrictions, type);
            do
            {
                function pfunc = new function(func.Func, restrictions, penaltymultiplier, n, type);
                curpoint = Gauss(pfunc, lastpoint, eps, maxiter);
                if ((curpoint - lastpoint).norm / curpoint.norm < eps)
                    flag = false;
                curpenalty = CalcPenalty(curpoint, restrictions, type);
                penaltymultiplier /= 2;
                k++;
                lastpoint = curpoint;
                Console.WriteLine($"{k} {curpoint} {curpenalty}");
            } while (k < maxiter && penaltymultiplier > eps && flag);
            return curpoint;
        }
        public static double CalcPenalty(Vector x, List<Restriction> restrictions, TypesOfPenalty type)
        {
            double res = 0;
            foreach (var item in restrictions)
            {
                switch (type)
                {
                    case TypesOfPenalty.Quadratic:
                        res += item.GetPenaltyQuard(x);
                        break;
                    case TypesOfPenalty.Linear:
                        res += item.GetPenaltyLinear(x);
                        break;
                    default:
                        break;
                }

            }
            return res;
        }
        public static double CalcPenalty(Vector x, List<UnequalityRestriction> restrictions, TypesOfBoundary type)
        {
            double res = 0;
            foreach (var item in restrictions)
            {
                switch (type)
                {
                    case TypesOfBoundary.Fractional:
                        res += item.GetBoundaryFrac(x);
                        break;
                    case TypesOfBoundary.Logarythmic:
                        res += item.GetBoundaryLog(x);
                        break;
                    default:
                        break;
                }

            }
            return res;
        }
        public static Vector SimpleRandomSearch(function func, Vector a, Vector b, double eps, double P)
        {
            int dim = a.dim;
            double V = 1;
            for (int i = 0; i < dim; i++)
            {
                V *= b.v[i] - a.v[i];
            }
            double Peps = Math.Pow(eps, dim) / V;
            long n = (long)Math.Ceiling(Math.Log(1 - P) / Math.Log(1 - Peps));
            Vector curpoint = new(dim);
            Vector res = new Vector(dim);
            double resval, curres;
            for (int i = 0; i < dim; i++)
            {
                res.v[i] = a.v[i] + rand.NextDouble() * (b.v[i] - a.v[i]);
            }

            resval = func.Func(curpoint);
            for (int k = 1; k < n; k++)
            {
                for (int i = 0; i < dim; i++)
                {
                    curpoint.v[i] = a.v[i] + rand.NextDouble() * (b.v[i] - a.v[i]);
                }
                curres = func.Func(curpoint);
                if (curres < resval)
                {
                    for (int i = 0; i < dim; i++)
                    {
                        res.v[i] = curpoint.v[i];
                    }
                    resval = curres;
                }
            }
            Console.WriteLine($"{eps} {P} {n} {res.v[0]} {res.v[1]} {-func.Func(res)}");
            return res;
        }
        public static Vector RandomSearch1(function func, Vector a, Vector b, double maxiter, double eps, Vector x0)
        {
            Vector respoint = Newton(func, x0, eps, 1000);
            double resvalue = func.Func(respoint);
            int k = 0;
            int n = a.dim;
            Vector curpoint = new(n);
            double curvalue;
            while (k < maxiter)
            {
                for (int i = 0; i < n; i++)
                {
                    curpoint.v[i] = a.v[i] + rand.NextDouble() * (b.v[i] - a.v[i]);
                }
                curpoint = Newton(func, curpoint, eps, 1000);
                curvalue = func.Func(curpoint);
                if (curvalue < resvalue)
                {
                    k = 0;
                    for (int i = 0; i < n; i++)
                    {
                        respoint.v[i] = curpoint.v[i];
                    }
                    resvalue = curvalue;
                }
                else
                {
                    k++;
                }
            }
            return respoint;
        }
        public static Vector RandomSearch2(function func, Vector a, Vector b, double maxiter, double eps, Vector x0)
        {
            Vector respoint = Newton(func, x0, eps, 1000);
            double resvalue = func.Func(respoint);
            int k = 0;
            int n = a.dim;
            Vector curpoint = new(n);
            double curvalue;
            while (k < maxiter)
            {
                for (int i = 0; i < n; i++)
                {
                    curpoint.v[i] = a.v[i] + rand.NextDouble() * (b.v[i] - a.v[i]);
                }
                curvalue = func.Func(curpoint);
                if (curvalue < resvalue)
                {
                    k = 0;
                    respoint = Newton(func, curpoint, eps, 1000);
                    resvalue = func.Func(respoint);
                }
                else
                {
                    k++;
                }
            }
            return respoint;
        }
        public static Vector RandomSearch3(function func, Vector x0, double maxiter, double eps)
        {
            Vector respoint = Newton(func, x0, eps, 1000);
            double resvalue = func.Func(respoint);
            int k = 0;
            int n = x0.dim;
            Vector curdirection = new(n);
            Vector curpoint = new(n);
            Vector newpoint = new(n);
            double curvalue, lastvalue = resvalue;
            while (k < maxiter)
            {
                for (int i = 0; i < n; i++)
                {
                    curdirection.v[i] = -1 + 2 * rand.NextDouble();
                }
                curpoint = respoint + curdirection;
                curvalue = func.Func(curpoint);
                int searchcounter = 0;
                curdirection *= 2;
                while (lastvalue < curvalue && searchcounter < maxiter)
                {
                    curpoint += curdirection;
                    lastvalue = curvalue;
                    curvalue = func.Func(curpoint);
                    curdirection *= 2;
                    searchcounter++;
                }
                if (searchcounter == maxiter)
                    k++;
                else
                {
                    newpoint = Newton(func, curpoint, eps, 1000);
                    if (func.Func(newpoint) < resvalue)
                    {
                        respoint = newpoint;
                        resvalue = func.Func(respoint);
                        k = 0;
                    }
                    else
                        k++;
                }
            }
            return respoint;
        }
        public static void SimplexMethod(int n, List<double> Q, int m, List<List<double>> x)
        {
            List<double> result = new List<double>();
            List<int> basis = new();
            List<int> nonbasis = new();
            for (int i = 0; i < m; i++)
            {
                basis.Add(n + i);
            }
            for (int i = 0; i < n; i++)
            {
                nonbasis.Add(i);
                result.Add(0);
            }
            for (int i = 0; i < m; i++)
            {
                result.Add(x[i][n]);
            }
            double[][] SimplexTable = new double[m + 1][];
            for (int i = 0; i < m + 1; i++)
            {
                SimplexTable[i] = new double[n + 1];
            }
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    SimplexTable[i][j] = -x[i][j];
                }
                SimplexTable[i][n] = x[i][n];
            }
            for (int j = 0; j < n; j++)
            {
                SimplexTable[m][j] = -Q[j];
            }
            SimplexTable[m][n] = Q[n];
            bool flag = true;
            while (flag)
            {
                PrintSimpex(n, m, SimplexTable, basis, nonbasis);
                flag = false;
                for (int i = 0; i < n; i++)
                {
                    if (SimplexTable[m][i] > 0)
                        flag = true;
                }
                if (flag == false)
                {
                    Console.WriteLine("Оптимальный план найден");
                }
                else
                {
                    Console.WriteLine("Выберите Разрешающий элемиент");
                    var str = Console.ReadLine().Split(' ');
                    int row = int.Parse(str[0]);
                    int column = int.Parse(str[1]);
                    JordanException(n, m, ref basis, ref nonbasis, ref SimplexTable, row, column);
                }
            }
        }
        public static void JordanException(int n, int m, ref List<int> basis, ref List<int> nonbasis, ref double[][] SimplexTable, int row, int column)
        {
            double EnablingElement = SimplexTable[row][column];
            SimplexTable[row][column] = 1 / EnablingElement;
            for (int i = 0; i < m + 1; i++)
            {
                if (i != row)
                {
                    SimplexTable[i][column] = -SimplexTable[i][column] / EnablingElement;
                }
            }
            for (int j = 0; j < n + 1; j++)
            {
                if (j != column)
                {
                    SimplexTable[row][j] /= EnablingElement;
                }
            }
            for (int i = 0; i < m + 1; i++)
            {
                for (int j = 0; j < n + 1; j++)
                {
                    if (i != row && j != column)
                    {
                        SimplexTable[i][j] += SimplexTable[i][column] * SimplexTable[row][j] * EnablingElement;
                    }
                }
            }
            int buf = basis[row];
            basis[row] = nonbasis[column];
            nonbasis[column] = buf;
        }
        static void PrintSimpex(int n, int m, double[][] SimplexTable, List<int> basis, List<int> nonbasis)
        {
            for (int i = 0; i < n; i++)
            {
                Console.Write($"-x{nonbasis[i] + 1} ");
            }
            Console.WriteLine("1");
            for (int i = 0; i < m + 1; i++)
            {
                if (i != m)
                    Console.Write($"x{basis[i] + 1} ");
                else
                    Console.Write("Q ");
                for (int j = 0; j < n + 1; j++)
                {
                    Console.Write($"{SimplexTable[i][j]:f3} ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("-----------");
            for (int i = 0; i < n + m; i++)
            {
                if (nonbasis.Contains(i))
                {
                    Console.Write($"{0:f3} ");
                }
                else
                {
                    Console.Write($"{SimplexTable[basis.IndexOf(i)][n]:f3} ");
                }
            }
            Console.WriteLine();
            Console.WriteLine("-----------");
        }
    }
    public class function
    {
        public enum OneDimensionSearch
        {
            GoldenRatio,
            QuadraticInterpollation
        }
        public OneDimensionSearch SearchType { get; set; } = OneDimensionSearch.GoldenRatio;
        public bool NumericDerivatives { get; set; } = false;
        public double diffeps { get; set; } = 1e-7;
        private Func<Vector, double> func;
        public int counter = 0;
        public double Func(Vector x)
        {
            counter++;
            return func(x);
        }
        public int dim { get; init; }
        List<Func<Vector, double>> gradlist;
        List<List<Func<Vector, double>>> GesseList;
        public Vector grad(Vector x0)
        {
            int n = dim;
            var res = new Vector(n);
            if (NumericDerivatives)
            {
                double f0 = 0, f1 = 0, delta = 0;
                f0 = Func(x0);
                for (int i = 0; i < n; i++)
                {
                    delta = Math.Abs(x0.v[i]) * diffeps;
                    x0.v[i] += delta;
                    f1 = Func(x0);
                    x0.v[i] -= delta;
                    res.v[i] = (f1 - f0) / delta;
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    res.v[i] = gradlist[i](x0);
                }
            }
            return res;
        }
        public Matrix Gesse(Vector x0)
        {
            int n = dim;
            var res = new Matrix(n);
            if (NumericDerivatives)
            {
                var delta = new Vector(n);
                for (int i = 0; i < n; i++)
                {
                    delta.v[i] = Math.Abs(x0.v[i]) * diffeps;
                }
                double f0, f1, f2, f3;
                f0 = Func(x0);
                for (int i = 0; i < n; i++)
                {
                    x0.v[i] += delta.v[i];
                    f1 = Func(x0);
                    x0.v[i] -= delta.v[i];
                    for (int j = 0; j < n; j++)
                    {
                        x0.v[j] += delta.v[j];
                        f2 = Func(x0);
                        x0.v[i] += delta.v[i];
                        f3 = Func(x0);
                        x0.v[j] -= delta.v[j];
                        x0.v[i] -= delta.v[i];
                        res.m[i][j] = (f3 - f2 - f1 + f0) / delta.v[i] / delta.v[j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        res.m[i][j] = GesseList[i][j](x0);
                    }
                }
            }

            return res;
        }
        public double DirectionMinimum(Vector x0, Vector dir)
        {
            int n = x0.dim;
            Func<double, double> function = (double lam) => Func(x0 + lam * dir);
            double res = 0;
            switch (SearchType)
            {
                case OneDimensionSearch.GoldenRatio:
                    double a, b;
                    Methods.findsectionwithminimum(function, out a, out b, 1, 1e-2, 10000);
                    res = Methods.Goldenratio(function, a, b, 1e-14);
                    break;
                case OneDimensionSearch.QuadraticInterpollation:
                    res = Methods.QuadraticInterpolation(function, 1, 1e-12);
                    break;
                default:
                    break;
            }

            return res;
        }

        public function(Func<Vector, double> func, int dim, List<Func<Vector, double>> gradlist, List<List<Func<Vector, double>>> gesseList)
        {
            this.func = func;
            this.dim = dim;
            this.gradlist = gradlist;
            GesseList = gesseList;
        }
        public function(Func<Vector, double> func, List<Restriction> penalties, double penaltymultiplier, int dim, TypesOfPenalty type)
        {
            this.dim = dim;
            gradlist = new();
            GesseList = new();
            this.func = (Vector x) =>
            {
                double res = func(x);
                foreach (var item in penalties)
                {
                    switch (type)
                    {
                        case TypesOfPenalty.Quadratic:
                            res += penaltymultiplier * item.GetPenaltyQuard(x);
                            break;
                        case TypesOfPenalty.Linear:
                            res += penaltymultiplier * item.GetPenaltyLinear(x);
                            break;
                        default:
                            break;
                    }

                }
                return res;
            };
        }
        public function(Func<Vector, double> func, List<UnequalityRestriction> penalties, double penaltymultiplier, int dim, TypesOfBoundary type)
        {
            this.dim = dim;
            gradlist = new();
            GesseList = new();
            this.func = (Vector x) =>
            {
                double res = func(x);
                foreach (var item in penalties)
                {
                    switch (type)
                    {
                        case TypesOfBoundary.Fractional:
                            res += penaltymultiplier * item.GetBoundaryFrac(x);
                            break;
                        case TypesOfBoundary.Logarythmic:
                            res += penaltymultiplier * item.GetBoundaryLog(x);
                            break;
                        default:
                            break;
                    }

                }
                return res;
            };
        }
        public function(Func<Vector, double> func, int dim)
        {
            this.func = func;
            this.dim = dim;
            gradlist = new();
            GesseList = new();
        }
    }
    public class Vector
    {
        public List<double> v;
        public Vector(int n)
        {
            v = new List<double>(n);
            for (int i = 0; i < n; i++)
            {
                v.Add(0);
            }
        }
        public Vector(int n, double val)
        {
            v = new List<double>(n);
            for (int i = 0; i < n; i++)
            {
                v.Add(val);
            }
        }
        public Vector(List<double> v)
        {
            this.v = v;
        }
        public int dim => v.Count;
        public double norm
        {
            get
            {
                double res = 0;
                foreach (var item in v)
                {
                    res += item * item;
                }
                return Math.Sqrt(res);
            }
        }
        public static double operator *(Vector a, Vector b)
        {
            if (a.dim != b.dim)
            {
                throw new Exception();
            }
            double res = 0;
            int n = a.dim;
            for (int i = 0; i < n; i++)
            {
                res += a.v[i] * b.v[i];
            }
            return res;
        }
        public static Vector operator *(double a, Vector b)
        {
            int n = b.dim;
            var res = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                res.v[i] = a * b.v[i];
            }
            return res;
        }
        public static Vector operator *(Vector a, double b)
        {
            int n = a.dim;
            var res = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                res.v[i] = b * a.v[i];
            }
            return res;
        }
        public static Vector operator +(Vector a, Vector b)
        {
            int n = a.dim;
            var res = new Vector(n);
            if (a.dim != b.dim)
            {
                throw new Exception();
            }
            for (int i = 0; i < n; i++)
            {
                res.v[i] = a.v[i] + b.v[i];
            }
            return res;
        }
        public static Vector operator -(Vector a, Vector b)
        {
            int n = a.dim;
            var res = new Vector(n);
            if (a.dim != b.dim)
            {
                throw new Exception();
            }
            for (int i = 0; i < n; i++)
            {
                res.v[i] = a.v[i] - b.v[i];
            }
            return res;
        }
        public static Vector operator -(Vector a)
        {
            int n = a.dim;
            var res = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                res.v[i] = -a.v[i];
            }
            return res;
        }
        public static Matrix operator ^(Vector a, Vector b)
        {
            int n = a.dim;
            var res = new Matrix(n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.m[i][j] = a.v[i] * b.v[j];
                }
            }
            return res;
        }
        public static Vector operator /(Vector a, double b)
        {
            int n = a.dim;
            var res = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                res.v[i] = a.v[i] / b;
            }
            return res;
        }
        public override string ToString()
        {
            StringBuilder res = new();
            for (int i = 0; i < dim - 1; i++)
            {
                res.Append($"{v[i]} ");
            }
            res.Append($"{v[dim - 1]}");
            return res.ToString();
        }
    }
    public class Matrix
    {
        public List<List<double>> m;
        public Matrix(int n)
        {
            m = new List<List<double>>(n);
            for (int i = 0; i < n; i++)
            {
                m.Add(new List<double>(n));
                for (int j = 0; j < n; j++)
                {
                    m[i].Add(0);
                }
            }
        }
        public Matrix(int n, double val)
        {
            m = new List<List<double>>(n);
            for (int i = 0; i < n; i++)
            {
                m.Add(new List<double>(n));
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                        m[i].Add(val);
                    else
                        m[i].Add(0);
                }
            }
        }
        public int dim => m.Count;
        public static Vector operator *(Matrix a, Vector b)
        {
            int n = a.dim;
            var res = new Vector(n);
            if (a.dim != b.dim)
                throw new Exception();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.v[i] += a.m[i][j] * b.v[j];
                }
            }
            return res;
        }
        public static Matrix operator *(Matrix a, Matrix b)
        {
            int n = a.dim;
            var res = new Matrix(n);
            if (a.dim != b.dim)
                throw new Exception();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        res.m[i][j] += a.m[i][k] * b.m[k][j];
                    }
                }
            }
            return res;
        }
        public static Matrix operator +(Matrix a, Matrix b)
        {
            int n = a.dim;
            var res = new Matrix(n);
            if (a.dim != b.dim)
                throw new Exception();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.m[i][j] += a.m[i][j] + b.m[i][j];
                }
            }
            return res;
        }
        public static Matrix operator -(Matrix a, Matrix b)
        {
            int n = a.dim;
            var res = new Matrix(n);
            if (a.dim != b.dim)
                throw new Exception();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.m[i][j] += a.m[i][j] - b.m[i][j];
                }
            }
            return res;
        }
        public Matrix Transpose()
        {
            int n = dim;
            var res = new Matrix(n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.m[i][j] = m[j][i];
                }
            }
            return res;
        }
        public static Matrix operator /(Matrix a, double b)
        {
            int n = a.dim;
            var res = new Matrix(n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    res.m[i][j] = a.m[i][j] / b;
                }
            }
            return res;
        }
    }
    public abstract class Restriction
    {
        protected Func<Vector, double> func;

        protected Restriction(Func<Vector, double> func)
        {
            this.func = func;
        }

        public abstract double GetPenaltyQuard(Vector x);
        public abstract double GetPenaltyLinear(Vector x);
    }
    public class EqualityRestriction : Restriction
    {
        public EqualityRestriction(Func<Vector, double> func) : base(func) { }

        public override double GetPenaltyLinear(Vector x)
        {
            return Math.Abs(func(x));
        }

        public override double GetPenaltyQuard(Vector x)
        {
            return func(x) * func(x);
        }
    }
    public class UnequalityRestriction : Restriction
    {
        public UnequalityRestriction(Func<Vector, double> func) : base(func) { }//g<=0

        public override double GetPenaltyLinear(Vector x)
        {
            return (func(x) + Math.Abs(func(x))) / 2.0;
        }

        public override double GetPenaltyQuard(Vector x)
        {
            return (func(x) + Math.Abs(func(x))) * (func(x) + Math.Abs(func(x))) / 4.0;
        }
        public double GetBoundaryLog(Vector x)
        {
            return -Math.Log(-func(x));
        }
        public double GetBoundaryFrac(Vector x)
        {
            return -1 / func(x);
        }
    }
}
