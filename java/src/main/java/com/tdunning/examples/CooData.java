package com.tdunning.examples;

import smile.math.matrix.SparseMatrix;

import java.util.Arrays;

/**
 * Data structure that is used to build a sparse matrix if given a bunch of i,j,x triples.
 * This just maintains arrays of i, j, and m[i,j], but has some cleverness about reallocating
 * as data arrives and about sorting into a good order for being a real SparseMatrix.
 * <p>
 * This is really inefficient for anything except accumulating entries. Any real processing will
 * need to be done by converting to csr or csc sparse formats. The SMILE SparseMatrix is a csc format.
 */
public class CooData {
    private int entriesAddedSinceCompression = 0;
    private ElementOrdering lastOrdering = ElementOrdering.NONE;

    int entries;
    private int nrows;
    private int ncols;
    int[] rows;
    int[] cols;
    double[] values;

    public CooData() {
        nrows = -1;
        ncols = -1;
        init(100, 100);
    }

    public CooData(int rows, int cols) {
        this.nrows = rows;
        this.ncols = cols;
        init(rows, cols);
    }

    private void init(int rows, int cols) {
        int n = Math.max(rows, cols) + 5;
        this.rows = new int[n];
        this.cols = new int[n];
        this.values = new double[n];
    }

    /**
     * Adds a value to the value already at i,j.
     *
     * @param i The row
     * @param j The column
     * @param x The increment to the value at A[i,j]
     */
    public void add(int i, int j, double x) {
        if (i < 0 || (nrows != -1 && i >= nrows)) {
            throw new IllegalArgumentException(String.format("Invalid row %d (should be in [0,%d)", i, nrows));
        }
        if (j < 0 || (ncols != -1 && i >= ncols)) {
            throw new IllegalArgumentException(String.format("Invalid row %d (should be in [0,%d)", j, ncols));
        }

        if (entries >= rows.length) {
            if (entriesAddedSinceCompression > entries / 4.0) {
                compress(ElementOrdering.BY_COL, false);
            }
            int n = 2 * entries;
            if (n > rows.length) {
                rows = Arrays.copyOf(rows, n);
                cols = Arrays.copyOf(cols, n);
                values = Arrays.copyOf(values, n);
            }
        }
        rows[entries] = i;
        cols[entries] = j;
        values[entries] = x;
        entries++;
        lastOrdering = ElementOrdering.NONE;
        entriesAddedSinceCompression++;
    }

    /**
     * Reorder and aggregate data and indexes to be a proper sparse matrix.
     *
     * @return The resulting matrix
     */
    public SparseMatrix asSparseMatrix() {
        compress(ElementOrdering.BY_COL, false);
        resolveSizing();

        // data is now sorted by col, then row
        // we just need to make a short column index
        // note that we create one last element to point to the end of all data
        int[] colIndex = new int[ncols + 1];
        int last = -1;
        int j = 0;
        for (int k = 0; k < entries; ) {
            assert rows[k] >= 0 && rows[k] < nrows;
            assert cols[k] >= 0 && cols[k] < ncols;

            while (j <= cols[k]) {
                colIndex[j++] = k;
            }
            last = cols[k];
            while (k < entries && cols[k] == last) {
                k++;
            }
        }
        colIndex[ncols] = entries;
        return new SparseMatrix(nrows, ncols, values, rows, colIndex);
    }

    private void resolveSizing() {
        if (ncols == -1 || nrows == -1) {
            for (int k = 0; k < entries; k++) {
                ncols = Math.max(ncols, cols[k] + 1);
                nrows = Math.max(nrows, rows[k] + 1);
            }
        }
    }

    enum ElementOrdering {
        NONE, BY_ROW, BY_COL
    }

    @SuppressWarnings("WeakerAccess")
    public void compress(ElementOrdering elementOrdering, boolean force) {
        if (!force && lastOrdering == elementOrdering) {
            return;
        }
        entriesAddedSinceCompression = 0;
        lastOrdering = elementOrdering;

        int[] major;
        int[] minor;
        switch (elementOrdering) {
            case BY_ROW:
                major = this.rows;
                minor = this.cols;
                break;
            case BY_COL:
            default:
                major = this.cols;
                minor = this.rows;
        }

        // first sort everything in row order
        int[] order = new int[entries];
        Sort.sort(order, major, 0, entries);
        untangle(order, major, 0, entries);
        untangle(order, minor, 0, entries);
        untangle(order, values, 0, entries);

        // now scan through all the data
        int fill = 0;
        for (int i = 0; i < entries; ) {
            // for each range of constant row number, sort by column
            int j = i + 1;
            while (j < entries && major[j] == major[i]) {
                j++;
            }
            if (j > i + 1) {
                Sort.sort(order, minor, i, j - i);
                untangle(order, major, i, j);
                untangle(order, minor, i, j);
                untangle(order, values, i, j);
            }

            // and now collapse ranges of constant column number
            for (int k = i; k < j; ) {
                int r = major[k];
                int c = minor[k];
                double sum = 0;
                for (; k < j && minor[k] == c; k++) {
                    sum += values[k];
                }
                major[fill] = r;
                minor[fill] = c;
                values[fill] = sum;
                fill++;
            }
            i = j;
        }

        entries = fill;
    }

    private void untangle(int[] order, int[] values, int start, int end) {
        int[] tmp = Arrays.copyOfRange(values, start, end);
        for (int i = start; i < end; i++) {
            tmp[i - start] = values[order[i]];
        }
        System.arraycopy(tmp, 0, values, start, end - start);
    }

    private void untangle(int[] order, double[] values, int start, int end) {
        double[] tmp = Arrays.copyOfRange(values, start, end);
        for (int i = start; i < end; i++) {
            tmp[i - start] = values[order[i]];
        }
        System.arraycopy(tmp, 0, values, start, end - start);
    }

    public void append(CooData other) {
        if (entries + other.entries > rows.length) {
            int n = entries + other.entries;
            rows = Arrays.copyOf(rows, n);
            cols = Arrays.copyOf(cols, n);
            values = Arrays.copyOf(values, n);
        }
        System.arraycopy(rows, entries, other.rows, 0, other.entries);
        System.arraycopy(cols, entries, other.cols, 0, other.entries);
        System.arraycopy(values, entries, other.values, 0, other.entries);
    }
}
