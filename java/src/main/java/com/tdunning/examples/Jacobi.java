package com.tdunning.examples;

import smile.math.matrix.SparseMatrix;

import java.util.Arrays;

/**
 * Classic iterative solver for sparse systems. This converges if the matrix A is diagonally dominant or if
 * it is symmetrical and positive definite.
 */
public class Jacobi {
    private SparseMatrix a;

    public Jacobi(SparseMatrix a) {
        if (a.ncols() != a.nrows()) {
            throw new IllegalArgumentException("Matrix must be square");
        }
        this.a = a;
    }

    public double[] solve(double[] b) {
        return solve(b, 1e-10, 10000);
    }

    public double[] solve(double[] b, double tolerance, int maxIteration) {
        final int n = a.ncols();
        if (b.length != n) {
            throw new IllegalArgumentException("Must have b vector same size as matrix");
        }

        double[] x = new double[n];
        double[] diagonal = new double[n];
        a.foreachNonzero((i, j, value) -> {
            if (i == j) {
                diagonal[i] = value;
            }
        });

        double dMax = Double.POSITIVE_INFINITY;
        int iteration = 0;
        while (dMax > tolerance && iteration < maxIteration) {
            // z = b - Rx, where R is A except for diagonal elements
            double[] tmp = Arrays.copyOf(b, n);
            a.foreachNonzero((i, j, value) -> {
                if (i != j) {
                    tmp[i] -= value * x[j];
                }
            });

            dMax = 0;
            for (int i = 0; i < n; i++) {
                double v = tmp[i] / diagonal[i];
                dMax = Math.max(Math.abs(x[i] - v), dMax);
                x[i] = v;
            }
            iteration++;
            System.out.printf("%10.2f\n", dMax);
        }
        return x;
    }
}
