using System;
using System.Collections.Generic;

namespace MetodiOptimizatsii
{
    class Program
    {
        static void Main(string[] args)
        {
            Func<Vector, double> z = (Vector x) => 100 * (x.v[1] - x.v[0]) * (x.v[1] - x.v[0]) + (1 - x.v[0]) * (1 - x.v[0]);
            List<Func<Vector, double>> grads = new();
            grads.Add((Vector x) => -200 * (x.v[1] - x.v[0]) - 2 * (1 - x.v[0]));
            grads.Add((Vector x) => 200 * (x.v[1] - x.v[0]));
            List<List<Func<Vector, double>>> Gesse = new();
            Gesse.Add(new List<Func<Vector, double>>());
            Gesse[0].Add((Vector x) => 202);
            Gesse[0].Add((Vector x) => -200);
            Gesse.Add(new List<Func<Vector, double>>());
            Gesse[1].Add((Vector x) => -200);
            Gesse[1].Add((Vector x) => 200);

            Func<Vector, double> z1 = (Vector x) => 100 * (x.v[1] - x.v[0] * x.v[0]) * (x.v[1] - x.v[0] * x.v[0]) + (1 - x.v[0]) * (1 - x.v[0]);
            List<Func<Vector, double>> grads1 = new();
            grads1.Add((Vector x) => -400 * (x.v[1] * x.v[0] - x.v[0] * x.v[0] * x.v[0]) - 2 * (1 - x.v[0]));
            grads1.Add((Vector x) => 200 * (x.v[1] - x.v[0] * x.v[0]));
            List<List<Func<Vector, double>>> Gesse1 = new();
            Gesse1.Add(new List<Func<Vector, double>>());
            Gesse1[0].Add((Vector x) => -400 * x.v[1] + 1200 * x.v[0] * x.v[0] + 2);
            Gesse1[0].Add((Vector x) => -400 * x.v[0]);
            Gesse1.Add(new List<Func<Vector, double>>());
            Gesse1[1].Add((Vector x) => -400 * x.v[0]);
            Gesse1[1].Add((Vector x) => 200);

            Func<Vector, double> fuck3 = (Vector x) =>
             {
                 return -(3 / (1 + (x.v[0] - 2) * (x.v[0] - 2) + (x.v[1] - 3) * (x.v[1] - 3) / 4) + 1 / (1 + (x.v[0] - 1) * (x.v[0] - 1) / 4 + (x.v[1] - 1) * (x.v[1] - 1)));
             };
            function f1 = new function(z, 2, grads, Gesse);
            f1.NumericDerivatives = false;
            function f2 = new function(z1, 2, grads1, Gesse1);
            f2.NumericDerivatives = false;
            function f3 = new function(z1, 2, null, null);
            f3.NumericDerivatives = true;
            var q = f3.grad(new Vector(2, 0.1));
            Vector x0 = new(2);
            x0.v[0] = 3;
            x0.v[1] = 4;

            var resfuck = Methods.DavidonFletcherPauel(f3, x0, 1e-1, 10000,1);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.DavidonFletcherPauel(f3, x0, 1e-3, 10000,1);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.DavidonFletcherPauel(f3, x0, 1e-5, 10000,1);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.DavidonFletcherPauel(f3, x0, 1e-7, 10000,1);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            Console.WriteLine("--------------");
            resfuck = Methods.Newton(f3, x0, 1e-1, 10000);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.Newton(f3, x0, 1e-3, 10000);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.Newton(f3, x0, 1e-5, 10000);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
            resfuck = Methods.Newton(f3, x0, 1e-7, 10000);
            Console.WriteLine($"{resfuck.v[0]} {resfuck.v[1]}");
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
        public static void findsectionwithminimum(Func<double, double> func, out double a, out double b, double x0, double delta)
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
            while (flag)
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
            if(NumericDerivatives)
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
            Func<double, double> function = (double lam) => func(x0 + lam * dir);
            double a, b;
            Methods.findsectionwithminimum(function, out a, out b, 1, dir.norm);
            return Methods.Goldenratio(function, a, b, 1e-14);
        }
        public function(Func<Vector, double> func, int dim, List<Func<Vector, double>> gradlist, List<List<Func<Vector, double>>> gesseList)
        {
            this.func = func;
            this.dim = dim;
            this.gradlist = gradlist;
            GesseList = gesseList;
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

}
