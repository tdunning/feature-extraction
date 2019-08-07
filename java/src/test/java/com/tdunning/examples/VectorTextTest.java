package com.tdunning.examples;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.junit.Test;
import smile.math.matrix.DenseMatrix;
import smile.math.matrix.JMatrix;
import smile.math.matrix.SparseMatrix;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.*;

public class VectorTextTest {
    private String sample = "We stayed for 5 nights last week. " +
            "The cold food of fruit/pastries/cereals is great. " +
            "$10 for a small OJ!! ";
    private String[] tokens = {
            "we", "stayed", "for", "5", "nights", "last", "week",
            "the", "cold", "food", "of", "fruit", "pastries",
            "cereals", "is", "great", "$10", "for", "a", "small", "oj"
    };

    @org.junit.Test
    public void tokenizeAsStream() {
        Stream<String> ref = Stream.of(tokens);
        Iterator<String> ix = ref.iterator();
        VectorText.tokenize(sample)
                .forEachOrdered(s -> assertEquals(ix.next(), s));
        assertFalse(ix.hasNext());
    }

    @org.junit.Test
    public void tokenize() {
        Iterator<String> is = Arrays.asList(tokens).iterator();
        for (String s : VectorText.tokenizeAsList(sample)) {
            assertEquals(is.next(), s);
        }
    }

    @Test
    public void vectorize() {
        // build a dictionary of all words we see in a subset of the text
        Map<String, Integer> dict = VectorText.tokenize(sample.split("\\.")[0]).collect(
                TreeMap::new,
                (Map<String, Integer> d, String s) -> d.put(s, d.size()),
                Map::putAll
        );
        dict.remove("5");
        dict.put("tiger", dict.size());

        int[] v = VectorText.vectorize(dict, sample);
        assertArrayEquals(new int[]{1, 1, 1, 0, 1, 1, 1}, v);
    }

    @Test
    public void count() {
        // build a dictionary of all words we see in a subset of the text
        Map<String, Integer> dict = VectorText.tokenize(sample.split("\\.")[0]).collect(
                TreeMap::new,
                (Map<String, Integer> d, String s) -> d.put(s, d.size()),
                Map::putAll
        );
        dict.remove("5");
        dict.put("tiger", dict.size());

        int[] v = VectorText.count(dict, sample);
        assertArrayEquals(new int[]{1, 1, 2, 0, 1, 1, 1}, v);
    }

    @Test
    public void gloveVectors() throws IOException {
        int nDocs = 50000;
        Progress p = new Progress();

        AtomicInteger docCount = new AtomicInteger();
        double t0 = System.nanoTime() / 1e9;
        Multiset<String> counts = docs(p, nDocs)
                .flatMap(VectorText::tokenize)
                .collect(
                        HashMultiset::create,
                        (strings, element) -> {
                            docCount.incrementAndGet();
                            strings.add(element);
                        },
                        HashMultiset::addAll);
        AtomicInteger wordCount = new AtomicInteger();
        Map<String, Integer> dict = counts.elementSet().stream()
                .filter(w -> counts.count(w) > 3)
                .collect(
                        TreeMap::new,
                        (d, w) -> d.put(w, wordCount.getAndIncrement()),
                        TreeMap::putAll);
        List<String> undict = new ArrayList<>(dict.keySet());

        Progress p1 = new Progress();
        DenseMatrix wordVectors = Files.lines(Paths.get("/Users/tdunning/Downloads/glove.6B/glove.6B.100d.txt"))
                .collect(
                        () -> new JMatrix(dict.size(), 100),
                        (DenseMatrix m, String rawWordVector) -> {
                            p1.log();
                            int i = rawWordVector.indexOf(' ');
                            String word = rawWordVector.substring(0, i);
                            if (dict.containsKey(word)) {
                                int row = dict.get(word);
                                int j = 0;
                                for (String v : rawWordVector.substring(i + 1).split(" ")) {
                                    try {
                                        m.set(row, j++, Double.parseDouble(v));
                                    } catch (NumberFormatException e) {
                                        System.out.printf("Error in %s\n%s\n", v, rawWordVector);
                                    }
                                }
                            }
                        },
                        DenseMatrix::add
                );

        DenseMatrix idf = new JMatrix(dict.size(), 1);
        for (String w : dict.keySet()) {
            idf.set(dict.get(w), 0, Math.log(counts.size() / counts.count(w)));
        }

        docs(p, 100)
                .forEach(
                        doc -> {
                            // for each document, build out sum of idf-weighted one-hot vectors
                            DenseMatrix docVector = new JMatrix(100, 1);
                            VectorText.tokenize(doc)
                                    .filter(dict::containsKey)
                                    .forEach(
                                            w -> {
                                                int iw = dict.get(w);
                                                for (int i = 0; i < 100; i++) {
                                                    docVector.add(i, 0, wordVectors.get(iw, i) * idf.get(iw, 0));
                                                }
                                            }
                                    );

                            // now multiply back at the word vectors to find nearest terms
                            // (dict.size() x 100) * (100 x 1) => (dict.size() x 1)
                            DenseMatrix r = wordVectors.abmm(docVector);

                            // find words with highest score
                            PriorityQueue<ScoredPair> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> a.score));
                            for (int i = 0; i < r.nrows(); i++) {
                                pq.add(new ScoredPair(i, 0, r.get(i, 0)));
                                while (pq.size() > 50) {
                                    pq.poll();
                                }
                            }

                            // reverse into descending order
                            List<Integer> best = pq.stream()
                                    .map(scoredItem -> scoredItem.i)
                                    .collect(Collectors.toList());
                            Collections.reverse(best);

                            // and let's take a look
                            System.out.printf("%s\n    ", doc.substring(0, Math.min(50, doc.length())));
                            for (Integer w : best) {
                                System.out.printf(" %s", undict.get(w));
                            }
                            System.out.printf("\n");
                        }
                );
    }

    private static class Progress {
        int step = 1;
        int order = 1;

        int count = 0;
        double t0 = System.nanoTime() * 1e-9;
        private int oldCount = 0;

        void log() {
            count++;
            if (count == step * order) {
                double t1 = System.nanoTime() * 1e-9;
                double rate = (count - oldCount) / (t1 - t0);
                t0 = t1;
                oldCount = count;
                System.out.printf("%d (%.0f / sec)\n", count, rate);

                step = (int) (2.51 * step);
                if (step > 10) {
                    step = 1;
                    order *= 10;
                }
            }
        }
    }

    @Test
    public void documentSpeed() throws IOException {
        int nDocs = -1;
        double frequencyCut = 1000;
        int minScore = 12;
        int maxAssociates = 100;

        double t0 = System.nanoTime() / 1e9;
        Progress p = new Progress();
        // count all the words in our corpus
        Multiset<String> counts = docs(p, nDocs)
                .flatMap(VectorText::tokenize)
                .collect(
                        HashMultiset::create,
                        Multiset::add,
                        HashMultiset::addAll);
        System.out.printf("%d total terms processed\n", counts.size());
        // build a dictionary with words that occur sufficiently
        Map<String, Integer> dict = counts.stream()
                .filter(w -> counts.count(w) > 3)
                .collect(
                        TreeMap::new,
                        (d, w) -> d.put(w, d.size()),
                        TreeMap::putAll);

        // invert our dictionary as well
        Map<Integer, String> undict = new HashMap<>();
        for (String w : dict.keySet()) {
            undict.put(dict.get(w), w);
        }


        double t1 = System.nanoTime() / 1e9;
        System.out.printf("built dictionaries %.1f MB/s\n", new File("/Users/tdunning/tmp/OpinRank/hotels.txt").length() / (t1 - t0) / 1e6);
        p = new Progress();

        Random rand = new Random();

        // print some documents out for reference and checking
        AtomicInteger id = new AtomicInteger(0);
        Map<Integer, Set<String>> ref = docs(p, 10)
                .collect(
                        TreeMap::new,
                        (m, raw) -> {
                            int currentDoc = id.getAndIncrement();
                            // downsample our words according to limit max frequency
                            // and translate to integer form
                            Set<String> words = VectorText.tokenize(raw)
                                    .filter(w -> dict.containsKey(w) && (rand.nextDouble() < frequencyCut / counts.count(w)))
                                    .map(w -> w + "-" + dict.get(w))
                                    .collect(Collectors.toSet());
                            m.put(currentDoc, words);
                        },
                        Map::putAll);

        for (Integer docId : ref.keySet()) {
            System.out.printf("%d: (", docId);
            for (String w : ref.get(docId)) {
                System.out.printf("%s ", w);
            }
            System.out.printf(")\n");
        }
        System.out.printf("\n");

        p = new Progress();

        // do the cooccurrence counting with downsampling of common items
        t0 = System.nanoTime() / 1e9;
        AtomicInteger docid = new AtomicInteger(0);
        CooData binaryTerms = docs(p, nDocs)
                .collect(
                        CooData::new,
                        (CooData m, String raw) -> {
                            int currentDoc = docid.getAndIncrement();
                            // downsample our words according to limit max frequency
                            // and translate to integer form
                            CooData words = VectorText.tokenize(raw)
                                    .filter(w -> dict.containsKey(w) && (rand.nextDouble() < frequencyCut / counts.count(w)))
                                    .collect(
                                            () -> m,
                                            (CooData mx, String w) -> mx.add(currentDoc, dict.get(w), 1.0),
                                            CooData::append);
                        },
                        CooData::append);
        binaryTerms.compress(CooData.ElementOrdering.BY_COL, false);
        for (int k = 0; k < binaryTerms.entries; k++) {
            binaryTerms.values[k] = 1;
        }
        t1 = System.nanoTime() / 1e9;
        System.out.printf("build doc matrix %.1f MB/s\n", new File("/Users/tdunning/tmp/OpinRank/hotels.txt").length() / (t1 - t0) / 1e6);

        SparseMatrix docByTerms = binaryTerms.asSparseMatrix();
        double[] finalCounts = new double[docByTerms.ncols()];
        docByTerms.foreachNonzero(
                (doc, word, k) -> {
                    finalCounts[word]++;
                });
        int totalDocuments = docByTerms.nrows();
        int totalWords = docByTerms.ncols();

        System.out.printf("doc matrix is %d x %d (%d vs %d non-zeros)\n", docByTerms.nrows(), docByTerms.ncols(), docByTerms.size(), binaryTerms.entries);
        SparseMatrix cooc = docByTerms.ata();
        System.out.printf("%d x %d (%d non-zeros)\n", cooc.nrows(), cooc.ncols(), cooc.size());

        // build associates matrix for words
        CooData rawConnections = new CooData(cooc.nrows(), cooc.ncols());
        for (int word = 0; word < totalWords; word++) {
            PriorityQueue<ScoredPair> highScores = new PriorityQueue<>(Comparator.comparingDouble(t12 -> t12.score));

            // scan through each column, scoring cooccurrences
            cooc.foreachNonzero(word, word + 1,
                    (w1, w2, k11) -> {
                        double k1x = finalCounts[w1];
                        double kx1 = finalCounts[w2];
                        double k12 = k1x - k11;
                        double k21 = kx1 - k11;
                        double k22 = totalDocuments - k11 - k12 - k21;
                        double score = llr(k11, k12, k21, k22);
                        if (score > minScore && (highScores.size() < maxAssociates || score > highScores.peek().score)) {
                            highScores.add(new ScoredPair(w1, w2, score));
                        }
                        while (highScores.size() > maxAssociates) {
                            highScores.poll();
                        }
                    });
            while (highScores.size() > 0) {
                ScoredPair associate = highScores.poll();
                rawConnections.add(associate.i, associate.j, 1);
            }
        }

        SparseMatrix associates = rawConnections.asSparseMatrix();
        SparseMatrix similar = associates.ata();
        for (String w : new String[]{"wild", "bad", "good", "lovely", "hotel", "rail"}) {
            System.out.printf("%s: ", w);
            similar.foreachNonzero(dict.get(w), dict.get(w) + 1,
                    (w1, w2, x) -> {
                        if (x > 8) {
                            System.out.printf("%s-%.0f ", undict.get(w1), x);
                        }
                    });
            System.out.printf("\n");
        }
    }

    private double h(double... kxx) {
        double sum = 0;
        for (double k : kxx) {
            sum += k;
        }
        double r = 0;
        for (double k : kxx) {
            if (k > 0) {
                r -= k * Math.log(k / sum);
            }
        }
        return r;
    }

    private double llr(double k11, double k12, double k21, double k22) {
        return 2 * (h(k11 + k12, k21 + k22) + h(k11 + k21, k12 + k22) - h(k11, k12, k21, k22));
    }

    @Test
    public void testHash() {
        int[] counts = new int[65536];
        for (int i = 0; i < 65536; i++) {
            for (int j = 0; j < 65536; j++) {
                int k = new IntPair(i, j).hashCode();
                k = k % counts.length;
                if (k < 0) {
                    k += counts.length;
                }
                counts[k]++;
            }
        }
        int[] tmp = Arrays.copyOf(counts, counts.length);
        Arrays.sort(tmp);

        double qSoFar = 0;
        System.out.printf("%10.3f %d\n", qSoFar, 0);
        for (double q = 0; q < 0.9; q += 0.1) {
            System.out.printf("%10.3f %d\n", q, tmp[(int) (q * tmp.length)]);
        }
        for (double q = 0.9; q < 0.99; q += 0.01) {
            System.out.printf("%10.3f %d\n", q, tmp[(int) (q * tmp.length)]);
        }
        for (double q = 0.99; q < 1; q += 0.001) {
            System.out.printf("%10.3f %d\n", q, tmp[(int) (q * tmp.length)]);
        }

        System.out.printf("\n\nbig\n");
        int last = 0;
        for (int i = 0; i < 65536; i++) {
            if (counts[i] >= 1082052) {
                System.out.printf("%10x %10d %10d\n", i, i, i - last);
                last = i;
            }
        }
        System.out.printf("\nend\n");
    }

    private class IntPair {
        public IntPair(int i, int j) {
            this.i = i;
            this.j = j;
        }

        int i, j;

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            IntPair intPair = (IntPair) o;
            return i == intPair.i &&
                    j == intPair.j;
        }

        @Override
        public int hashCode() {
            int seed = 3;
            // murmur is nice for general bit mixing, but it has some nasty favored patterns
            return 1037 * murmur(seed, i, j) + 17 * i + 53 * j;
        }

        private int murmur(int seed, int i, int j) {
            // one round of murmur
            int c1 = 0xcc9e2d51;
            int c2 = 0x1b873593;

            int k = i;
            k *= c1;
            k = (k << 15) | (k >> 17);
            k *= c2;

            int h = seed ^ k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;

            k = j;
            k *= c1;
            k = (k << 15) | (k >> 17);
            k *= c2;
            h = h ^ k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;

            h ^= h >>> 16;
            h *= 0x85ebca6b;
            h ^= h >>> 13;
            h *= 0xc2b2ae35;
            h ^= h >>> 16;

            return h;
        }
    }

    private Stream<String> docs(Progress p) throws IOException {
        return docs(p, -1);
    }

    private Stream<String> docs(Progress p, int limit) throws IOException {
        Function<String, String> parser = line -> {
            p.log();
            int k = line.indexOf('\t');
            if (k >= 0) {
                k = line.indexOf('\t', k + 1);
                if (k >= 0) {
                    return line.substring(k + 1);
                } else {
                    throw new IllegalArgumentException("Couldn't find second tab");
                }
            } else {
                throw new IllegalArgumentException("Couldn't find first tab");
            }
        };
        if (limit <= 0) {
            return Files.lines(Paths.get("/Users/tdunning/tmp/OpinRank/hotels.txt"), StandardCharsets.ISO_8859_1)
                    .map(parser);
        } else {
            return Files.lines(Paths.get("/Users/tdunning/tmp/OpinRank/hotels.txt"), StandardCharsets.ISO_8859_1)
                    .limit(limit)
                    .map(parser);

        }
    }

    private class ScoredPair {
        private final int i;
        private final int j;
        private final double score;

        public ScoredPair(int i, int j, double score) {
            this.i = i;
            this.j = j;
            this.score = score;
        }

        public int getI() {
            return i;
        }

        public int getJ() {
            return j;
        }

        public double getScore() {
            return score;
        }
    }
}