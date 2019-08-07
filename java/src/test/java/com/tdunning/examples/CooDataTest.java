package com.tdunning.examples;

import org.junit.Test;
import smile.math.matrix.SparseMatrix;

import java.util.*;

import static org.junit.Assert.*;

public class CooDataTest {

    public static final int ROWS = 100000;
    public static final int COLS = 100000;

    @Test
    public void basics() {
        Random rand = new Random();
        CooData m = new CooData(ROWS, COLS);
        Map<Pair, Double> ref = new TreeMap<>();
        for (int step = 0; step < 100000; step++) {
            int i;
            int j;

            if (rand.nextDouble() < 0.2) {
                i = rand.nextInt(20);
                j = rand.nextInt(20);
            } else {
                i = (int) (-10000 * Math.log(rand.nextDouble()));
                if (i >= ROWS) {
                    i = ROWS - 1;
                }
                j = (int) (-10000 * Math.log(rand.nextDouble()));
                if (j >= COLS) {
                    j = COLS - 1;
                }
            }
            double x = rand.nextGaussian();
            Pair k = new Pair(i, j);
            if (ref.containsKey(k)) {
                ref.put(k, ref.get(k) + x);
            } else {
                ref.put(k, x);
            }
            m.add(i, j, x);
        }
        m.compress(CooData.ElementOrdering.BY_ROW, true);

        for (int step = 0; step < m.entries; step++) {
            assertEquals(ref.get(new Pair(m.rows[step], m.cols[step])), m.values[step], 0);
        }
    }

    @Test
    public void small() {
        CooData m = new CooData(5, 7);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7; j++) {
                m.add(i, j, 100 * i + j);
            }
        }
        for (int i = 0; i < 5; i++) {
            m.add(i, i, 2);
            m.add(i, i + 1, 3);
        }
        m.compress(CooData.ElementOrdering.BY_ROW, true);
        int k = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 7; j++) {
                assertEquals(m.rows[k], i);
                assertEquals(m.cols[k], j);
                assertEquals(m.values[k], 100 * i + j + ((i == j) ? 2 : 0) + ((j == i + 1) ? 3 : 0), 0);
                k++;
            }
        }

        SparseMatrix mx = m.asSparseMatrix();
        k = 0;
        for (int j = 0; j < 7; j++) {
            for (int i = 0; i < 5; i++) {
                assertEquals(m.rows[k], i);
                assertEquals(m.cols[k], j);
                assertEquals(m.values[k], 100 * i + j + ((i == j) ? 2 : 0) + ((j == i + 1) ? 3 : 0), 0);
                assertEquals(m.values[k], mx.get(i,j), 0);
                k++;
            }
        }


    }

    private class Pair implements Comparable<Pair> {
        int i, j;

        public Pair(int i, int j) {
            this.i = i;
            this.j = j;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return i == pair.i &&
                    j == pair.j;
        }

        @Override
        public int hashCode() {
            return 31 * i + j;
        }

        @Override
        public int compareTo(Pair other) {
            int r = this.i - other.j;
            if (r == 0) {
                return this.j - other.j;
            } else {
                return r;
            }
        }
    }
}