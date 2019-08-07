package com.tdunning.examples;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class VectorText {
    private static Pattern word = Pattern.compile(String.join("", "",
//            "([A-Z]\\.)+",                                  // a word
            "\\d+:(\\.\\d)+",                              // a number
            "|(\\w+)",                                      // a word
            "|(https?://)?(\\w+\\.)(\\w{2,})+([\\w/]+)?",   // URL
            "|[@#]?\\w+(?:[-']\\w+)*",                      // twitter-like user reference
            "|\\$\\d+(\\.\\d+)?%?",                         // dollar amount
            "|\\\\[Uu]\\w+",                                // normal word
            "|\\\\[Uu]\\w+'t"                               // contraction
    ));

    @SuppressWarnings("WeakerAccess")
    public static Stream<String> tokenize(CharSequence s) {
        Iterator<String> is = new Iterator<String>() {
            int position = 0;
            Matcher m = word.matcher(s);

            @Override
            public boolean hasNext() {
                return m.find(position);
            }

            @Override
            public String next() {
                position = m.end();
                return m.group().toLowerCase();
            }
        };
        int characteristics = Spliterator.DISTINCT | Spliterator.SORTED | Spliterator.IMMUTABLE;
        Spliterator<String> spliterator = Spliterators.spliteratorUnknownSize(is, characteristics);

        return StreamSupport.stream(spliterator, false);
    }

    @SuppressWarnings("WeakerAccess")
    public static List<String> tokenizeAsList(CharSequence s) {
        return tokenize(s).collect(Collectors.toList());
    }

    public static int[] vectorize(Map<String, Integer> dictionary, String s) {
        int[] result = new int[dictionary.size()];
        VectorText.tokenize(s).forEach(w -> {
            if (dictionary.containsKey(w)) {
                result[dictionary.get(w)] = 1;
            }
        });
        return result;
    }

    public static int[] count(Map<String, Integer> dictionary, String s) {
        int[] result = new int[dictionary.size()];
        VectorText.tokenize(s).forEach(w -> {
            if (dictionary.containsKey(w)) {
                result[dictionary.get(w)]++;
            }
        });
        return result;
    }
}
