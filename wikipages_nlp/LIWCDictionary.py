import string
import re
import liwc


# #
# Interface to the LIWC dictionary, implementing patterns for each LIWC
# category based on the LIWC.cat file
#
# based on the Java implementation by Francois Mairesse
# http://www.mairesse.co.uk
# #

class LIWCDictionary:

    # #
    # Loads dictionary from LIWC dictionary tab-delimited text file
    # (with variable names as first row). Each word category is converted
    # into a regular expression that is a disjunction of all its members.
    #
    # @param cat_file
    #           dictionary file, it should be pointing to the LIWC.cat
    #           file of the Linguistic Inquiry and Word Count software
    #           (Pennebaker & Francis, 2001).
    # #
    def __init__(self, cat_file):
        # !!!!!!!!!!!!!!!!!! exception handling !!!!!!!!!!!!!
        try:
            # self.map = self.load_liwc_dictionary(cat_file)
            self.parse, self.category_names = liwc.load_token_parser(cat_file)
        except IOError:
            print("Cannot open the file: " + cat_file)

    # #
    # Loads dictionary from LIWC dictionary tab-delimited text file
    # (with variable names as first row). Each word category is converted
    # into a regular expression that is a disjunction of all its members.
    #
    # @param dic_file
    #           dictionary file, it should be pointing to the LIWC.cat
    #           file of the Linguistic Inquiry and Word Count software
    #           (Pennebaker & Francis, 2001).
    # @return dictionary list associating each category with a regular expression
    #           object matching each word in the category
    # #
    def load_liwc_dictionary(self, dic_file):
        word_map = {}
        current_cat = ""
        cat_regex = ""

        with open(dic_file) as fp:
            for line in fp:
                line_stripped = re.sub('\n', '', line)
                var_pattern = re.compile("\t[\w ]+")
                word_pattern = re.compile("\t\t.+ \(\d+\)")
                # if encounter new variable
                if var_pattern.match(line_stripped):
                    # add full regex to word_map
                    if cat_regex != "":
                        length = len(cat_regex)
                        cat_regex = cat_regex[:length - 1]
                        cat_regex = "(" + cat_regex + ")"
                        cat_regex = string.replace(cat_regex, "*", "[\\w\']*")
                        word_map[current_cat] = re.compile(cat_regex)
                    # update current category
                    current_cat = string.split(line_stripped, "\t")[1]
                    current_cat = current_cat.strip()
                    cat_regex = ""
                # if encounter new word
                elif word_pattern.match(line_stripped):
                    new_pattern = re.split("\s+", line_stripped)[1]
                    new_pattern = string.lower(new_pattern)
                    cat_regex += "\\b" + new_pattern + "\\b|"

        # add last regex to word_map
        if cat_regex != "":
            length = len(cat_regex)
            cat_regex = cat_regex[:length - 1]
            cat_regex = "(" + cat_regex + ")"
            cat_regex = string.replace(cat_regex, "*", "[\\w\']*")
            word_map[current_cat] = re.compile(cat_regex)

        return word_map

    # #
    # Returns a dictionary list associating each LIWC category to the number of
    # their occurrences in the input text. The counts are computed matching
    # patterns loaded.
    #
    # @param text
    #           input text
    # @return
    #           dictionary list associating each LIWC category with the percentage
    #           of words in the text belonging to it
    # #
    def get_counts(self, text):
        counts = {}
        words = self.tokenize(text)
        word_count = len(words)
        sentences = self.split_sentences(text)
        counts["WC"] = len(words)
        counts["WPS"] = 1.0 * len(words) / len(sentences)
        # count words with more than 6 letters and numbers
        six_letters = 0
        numbers = 0
        num_pattern = re.compile("-?[,\d+]*\.?\d+")
        for word in words:
            lc_word = word.lower()
            if len(lc_word) > 6:
                six_letters += 1
            if num_pattern.match(lc_word):
                numbers += 1

        # count unique words
        unique = list(set(words))
        counts["UNIQUE"] = 100.0 * len(unique) / word_count
        counts["SIXLTR"] = 100.0 * six_letters / word_count
        # count abbreviations
        abbrev = self.count_matches("\w\.(\w\.)+", text)
        counts["ABBREVIATIONS"] = 100.0 * abbrev / word_count
        # count emoticons
        emoticons = self.count_matches("[:;8%]-?[\)\(\@\[\]\|]+", text)
        counts["EMOTICONS"] = 100.0 * emoticons / word_count
        # count text ending with a question mark
        qmarks = self.count_matches("\w\s*\?", text)
        counts["QMARKS"] = 100.0 * qmarks / len(sentences)
        # count punctuation marks
        periods = self.count_matches("\.", text)
        counts["PERIOD"] = 100.0 * periods / word_count
        commas = self.count_matches(",", text)
        counts["COMMA"] = 100.0 * commas / word_count
        colons = self.count_matches(":", text)
        counts["COLON"] = 100.0 * colons / word_count
        semicolons = self.count_matches(";", text)
        counts["SEMIC"] = 100.0 * semicolons / word_count
        qmark = self.count_matches("\?", text)
        counts["QMARK"] = 100.0 * qmark / word_count
        exclams = self.count_matches("!", text)
        counts["EXCLAM"] = 100.0 * exclams / word_count
        dashes = self.count_matches("-", text)
        counts["DASH"] = 100.0 * dashes / word_count
        quotes = self.count_matches("\"", text)
        counts["QUOTE"] = 100.0 * quotes / word_count
        apostros = self.count_matches("\'", text)
        counts["APOSTRO"] = 100.0 * apostros / word_count
        parenths = self.count_matches("[\(\[{]", text)
        counts["PARENTH"] = 100.0 * parenths / word_count
        otherp = self.count_matches("[^\w\d\s\.:;\?!\"\'\(\{\[,-]", text)
        counts["OTHERP"] = 100.0 * otherp / word_count
        allp = periods + commas + colons + semicolons + qmark + exclams + dashes + quotes + apostros + parenths + otherp
        counts["ALLPCT"] = 100.0 * allp / word_count

        # initialize in_dic list to false
        in_dic = []
        for index in range(word_count):
            in_dic.append(False)

        # get all lexical counts
        for cat, pattern in self.map.iteritems():
            cat_count = 0
            for i, word in enumerate(words):
                lc_word = word.lower()
                if pattern.match(lc_word):
                    cat_count += 1
                    in_dic[i] = True
            counts[cat] = 100.0 * cat_count / word_count

        # get ratio of words matched
        words_matched = 0
        for index in range(len(in_dic)):
            if in_dic[index]:
                words_matched += 1
        counts["DIC"] = 100.0 * words_matched / word_count

        # add numbers
        counts["NUMBERS"] = 100.0 * numbers / word_count

        return counts



    # #
    # Splits a text into words separated by non-word characters
    #
    # @param text
    #           text to tokenize
    # @return
    #           a list of words
    # #
    def tokenize(self, text):
        words_only = re.sub("\W+\s*", " ", text)
        words_only = re.sub("\s+$", "", words_only)
        words_only = re.sub("^\s+", "", words_only)
        words = re.split("\s+", words_only)
        return words

    # #
    # Splits a text into sentences separated by a dot, exclamation point,
    # or question mark.
    #
    # @param text
    #           text to split
    # @return
    #           a list of sentences
    # #
    def split_sentences(self, text):
        sentences = re.split("\s*[.!?]+\s*", text)
        count = len(sentences)
        if not sentences[count - 1]:
            del sentences[count - 1]
        return sentences

    # #
    # Counts the number of times a pattern appears in a string
    #
    # @param regex
    #           regular expression string to be matched, it must be in
    #           the appropriate format to be compiled into a regular
    #           expression object
    # @param text
    #           input text
    # @return
    #           number of matches found
    # #
    def count_matches(self, regex, text):
        pattern = re.compile(regex)
        return len(re.findall(pattern, text))

    # #
    # Removes the HTML tags in an input string
    # CAUTION: this has XML vulnerabilities to maliciously constructed data!!
    #
    # @param html_string
    #           input text which possibly contains some html tags
    #
    # @return no_html_string
    #           the text that was input, stripped of html tags
    # #
    def remove_html(self, html_string):
        #return ''.join(xml.etree.ElementTree.fromstring(html_string).itertext())
        return re.sub('<[^<]+?>', '', html_string)  # strip html tags
