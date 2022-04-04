using System;
using System.Collections.Generic;
using System.Text;

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
            //List<Restriction> rests = new();
            List<UnequalityRestriction> rests = new();
            rests.Add(new UnequalityRestriction((Vector x) => x.v[0] + x.v[1] + 1));
            //rests.Add(new EqualityRestriction((Vector x) => x.v[1] - x.v[0] - 1));
            Func<Vector, double> ffunkk = (Vector x) => 4 * (x.v[1] - x.v[0]) * (x.v[1] - x.v[0]) + 3 * (x.v[0] - 1) * (x.v[0] - 1);
            function func = new function(ffunkk, 2);
            Vector x0 = new Vector(2, 0);
            x0.v[0] = -100;
            x0.v[1] = -100;
            //var res = Methods.PenaltyFunctions(func, x0, rests, 1e-6, 10000, TypesOfPenalty.Quadratic);//-1 0
            var res = Methods.BoudaryFunctions(func, x0, rests, 1e-15, 10000, TypesOfBoundary.Fractional);//-0.3 -0.7
            Console.WriteLine("Hello World!");
        }
    }
    public static class Methods
    {
        public static double Goldenratio(Func<double, double> func, double a, double b, double eps)
        {
            double k1 = (3 - Math.Sqrt(5)) / 2;
            double k2 = (Math.Sqrt(5) - 1) / 2;
            bool flag = true;
            double x1 = 0, x2 = 0;
            int i = 1;
            int lastchosen = 0;
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
        public static void findsectionwithminimum(Func<double, double> func, out double a, out double b, double x0, double delta, int maxiter)
        {
            a = 0;
            b = 0;
            double x1 = x0 + delta;
            double x2, f0, f1, f2;
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
                }
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
            while (b.norm > eps && k < maxiter)
            {
                SolveSlae(A, b);
                x0 = x0 + b * func.DirectionMinimum(x0, b);
                k++;
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
            Vector lastgrad = func.grad(x0);
            Vector xlast = x0;
            Vector deltax = omega * (Eta * lastgrad);
            double lambda = func.DirectionMinimum(x0, deltax);
            Vector curx = x0 + lambda * deltax;
            deltax *= lambda;
            Vector curgrad = func.grad(curx);
            Vector deltagrad = curgrad - lastgrad;
            k++;
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
            }
            Console.WriteLine(k);
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
            double fcur = func.func(x0);
            int n = func.dim;
            Vector curpoint = x0;
            do
            {
                flast = fcur;
                for (int i = 0; i < n; i++)
                {
                    Vector dir = new Vector(n, 0);
                    dir.v[i] = 1;
                    curpoint = curpoint + func.DirectionMinimum(curpoint, dir) * dir;
                }
                fcur = func.func(curpoint);
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
                function pfunc = new function(func.func, restrictions, penaltymultiplier, n, penaltytype);
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
            Vector curpoint = x0;
            int k = 0;
            int n = x0.dim;
            double curpenalty = CalcPenalty(curpoint, restrictions, type);
            do
            {
                function pfunc = new function(func.func, restrictions, penaltymultiplier, n, type);
                curpoint = Gauss(pfunc, curpoint, eps, maxiter);
                curpenalty = CalcPenalty(curpoint, restrictions, type);
                penaltymultiplier /= 2;
                k++;
                Console.WriteLine($"{k} {curpoint} {curpenalty}");
            } while (k < maxiter);
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
    }
    public class function
    {
        public bool NumericDerivatives { get; set; } = false;
        public double diffeps { get; set; } = 1e-7;
        public Func<Vector, double> func;
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
                f0 = func(x0);
                for (int i = 0; i < n; i++)
                {
                    delta = Math.Abs(x0.v[i]) * diffeps;
                    x0.v[i] += delta;
                    f1 = func(x0);
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
                f0 = func(x0);
                for (int i = 0; i < n; i++)
                {
                    x0.v[i] += delta.v[i];
                    f1 = func(x0);
                    x0.v[i] -= delta.v[i];
                    for (int j = 0; j < n; j++)
                    {
                        x0.v[j] += delta.v[j];
                        f2 = func(x0);
                        x0.v[i] += delta.v[i];
                        f3 = func(x0);
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
            Func<double, double> function = (double lam) => func(x0 + lam * dir / dir.norm);
            double a, b;
            Methods.findsectionwithminimum(function, out a, out b, 1, 1e-7, 10000);
            return Methods.Goldenratio(function, a, b, 1e-14);
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
