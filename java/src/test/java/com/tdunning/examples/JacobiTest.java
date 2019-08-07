package com.tdunning.examples;

import org.junit.Test;
import smile.math.matrix.SparseMatrix;

import static org.junit.Assert.*;

public class JacobiTest {

    @Test
    public void solve() {
        CooData connectionData = new CooData();
        // 100 x 100 mesh encoded as 10,000 elements in a vector
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                int k0 = coord(i, j);
                double sum = 0;
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        if ((dx != 0 || dy != 0) && i + dx >= 0 && i + dx < 100 && j + dy >= 0 && j + dy < 100) {
                            double w = 0.125;
                            sum += w;
                            connectionData.add(k0, coord(i + dx, j + dy), w);
                        }
                    }
                }
                connectionData.add(k0, k0, -sum);
            }
        }

        SparseMatrix transfer = connectionData.asSparseMatrix();

        Jacobi jSolver = new Jacobi(transfer);
        double[] b = new double[10000];
        for (int i = 0; i < 100; i++) {
            b[coord(i, 0)] = 1;
            b[coord(i, 99)] = -1;
        }
        double[] x = jSolver.solve(b);
        for (int j = 0; j < 10; j++) {
            System.out.printf("%.2f ", x[j]);
        }
        for (int j = 10; j < 100; j += 5) {
            System.out.printf("%.2f ", x[j]);
        }
        System.out.printf("\n");

    }

    private int coord(int i, int j) {
        return 100 * i + j;
    }
}
