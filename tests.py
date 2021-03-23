import unittest

import string
from Project4.Trie import Trie, TrieNode, TrieClassifier


class TestProject4(unittest.TestCase):

    def test_node_empty(self):
        """
        Test TrieNode.empty method
        """
        # empty node
        node = TrieNode()
        self.assertTrue(node.empty())

        # nonempty node
        node.children[0] = TrieNode()
        self.assertFalse(node.empty())

    def test_node_get_index(self):
        """
        Test TrieNode._get_index static method
        """
        # test lowercase letters
        for i, letter in enumerate(string.ascii_lowercase):
            index = TrieNode._get_index(letter)
            self.assertEqual(i, index)

        # test uppercase letters
        for i, letter in enumerate(string.ascii_uppercase):
            index = TrieNode._get_index(letter)
            self.assertEqual(i, index)

    def test_node_get_child(self):
        """
        Test TrieNode.get_child method
        """
        # check on node with no children
        node = TrieNode()
        for letter in string.ascii_lowercase:
            self.assertIsNone(node.get_child(letter))

        # add children and check on each one
        children = [TrieNode() for i in range(len(string.ascii_lowercase))]
        node.children = children
        for i, letter in enumerate(string.ascii_lowercase):
            result = node.get_child(letter)
            self.assertIs(result, node.children[i])

    def test_node_set_child(self):
        """
        Test TrieNode.set_child method
        """
        # set children and check they are added properly
        node = TrieNode()
        for i, letter in enumerate(string.ascii_lowercase):
            node.set_child(letter)
            self.assertIsNotNone(node.children[i])
            self.assertIsInstance(node.children[i], TrieNode)
            self.assertTrue(node.children[i].empty())


    def test_node_delete_child(self):
        """
        Test TrieNode.delete_child method
        """
        # add children
        node = TrieNode()
        children = [TrieNode() for i in range(len(string.ascii_lowercase))]
        node.children = children

        # ensure all children are deleted
        for i, letter in enumerate(string.ascii_lowercase):
            self.assertIsNotNone(node.children[i])
            node.delete_child(letter)
            self.assertIsNone(node.children[i])

        self.assertTrue(node.empty())

    def test_add(self):
        """
        Test Trie.add method
        """
        # short word
        trie = Trie()
        word = "msu"
        count = trie.add(word)

        # ensure add returns number of occurrences and size, unique are incremented
        self.assertEqual(count, 1)
        self.assertEqual(trie.size, 1)
        self.assertEqual(trie.unique, 1)

        # ensure m, s and u nodes are added properly
        m_child = trie.root.get_child(word[0])
        self.assertIsInstance(m_child, TrieNode)
        self.assertEqual(m_child.is_end, 0)
        s_child = m_child.get_child(word[1])
        self.assertIsInstance(s_child, TrieNode)
        self.assertEqual(s_child.is_end, 0)
        u_child = s_child.get_child(word[2])
        self.assertIsInstance(u_child, TrieNode)
        self.assertEqual(u_child.is_end, 1)

        # duplicate word
        count = trie.add(word)

        # ensure add returns number of occurrences, size IS incremented and unique is NOT
        self.assertEqual(count, 2)
        self.assertEqual(trie.size, 2)
        self.assertEqual(trie.unique, 1)

        # long word
        word = "algorithm"
        count = trie.add(word)

        # ensure add returns number of occurrences and size, unique are incremented
        self.assertEqual(count, 1)
        self.assertEqual(trie.size, 3)
        self.assertEqual(trie.unique, 2)

        # ensure nodes are configured properly
        child = trie.root
        for i, letter in enumerate(word):
            self.assertIsInstance(child, TrieNode)
            self.assertEqual(child.is_end, 0)
            child = child.get_child(letter)
        self.assertEqual(child.is_end, 1)       # bottom-updating loop means we are now at the final child

        # duplicate algorithm
        count = trie.add(word)
        self.assertEqual(count, 2)
        self.assertEqual(trie.size, 4)
        self.assertEqual(trie.unique, 2)

        # long word with shared stem
        word = "algorithmic"
        count = trie.add(word)

        # ensure add returns number of occurrences and size, unique are incremented
        self.assertEqual(count, 1)
        self.assertEqual(trie.size, 5)
        self.assertEqual(trie.unique, 3)

        # ensure nodes are configured properly
        child = trie.root
        for i, letter in enumerate(word):
            self.assertIsInstance(child, TrieNode)
            if i == len("algorithm"):               # catch algorithM (which is duplicated)
                self.assertEqual(child.is_end, 2)
            else:
                self.assertEqual(child.is_end, 0)
            child = child.get_child(letter)
        self.assertEqual(child.is_end, 1)       # bottom-updating loop means we are now at the final child

        # duplicate algorithmic
        count = trie.add(word)

        # ensure add returns number of occurrences, size IS incremented and unique is NOT
        self.assertEqual(count, 2)
        self.assertEqual(trie.size, 6)
        self.assertEqual(trie.unique, 3)

        # add several words
        words = "on the banks of red cedar theres a school thats known to all".split()
        size, unique = 6, 3
        for word in words:
            count = trie.add(word)
            size += 1
            unique += 1

            # ensure add returns number of occurrences and size, unique are incremented
            self.assertEqual(count, 1)
            self.assertEqual(trie.size, size)
            self.assertEqual(trie.unique, unique)

            # ensure nodes are configured properly
            child = trie.root
            for i, letter in enumerate(word):
                self.assertIsInstance(child, TrieNode)
                child = child.get_child(letter)
            self.assertEqual(child.is_end, 1)  # bottom-updating loop means we are now at the final child

    def test_search(self):
        """
        Test Trie.search method
        """
        # construct Trie
        trie = Trie()
        words = "on the banks of red cedar theres school thats known to all".split()
        for word in words:
            trie.add(word)

        # search for existing and non-existing words
        anti_words = [word[::-1] for word in words]
        for word, anti_word in zip(words, anti_words):
            count = trie.search(word)
            self.assertEqual(count, 1)
            count = trie.search(anti_word)
            self.assertEqual(count, 0)

        # add subset of duplicates and ensure proper count is returned
        duplicates = words[:len(words) // 2]
        for word in duplicates:
            trie.add(word)
        for word, anti_word in zip(words, anti_words):
            count = trie.search(word)       # search legitimate word
            if word in duplicates:
                self.assertEqual(count, 2)
            else:
                self.assertEqual(count, 1)
            count = trie.search(anti_word)  # search nonexistent word
            self.assertEqual(count, 0)

    def test_delete(self):
        """
        Test Trie.delete method
        """
        # construct Trie
        trie = Trie()
        trie.add("a")
        trie.add("a")
        trie.add("ab")
        trie.add("abc")

        # delete a, ensure count is returned, ensure size, unique are updated
        count = trie.delete("a")
        self.assertEqual(count, 2)
        self.assertEqual(trie.size, 2)
        self.assertEqual(trie.unique, 2)

        # ensure other words remain intact
        count = trie.search("ab")
        self.assertEqual(count, 1)
        count = trie.search("abc")
        self.assertEqual(count, 1)

        # ensure deletions on nonexistent nodes return zero and do not change size, unique
        count = trie.delete("msu")
        self.assertEqual(count, 0)
        self.assertEqual(trie.size, 2)
        self.assertEqual(trie.unique, 2)

        # delete abc, ensure count is returned, ensure size, unique are updated
        count = trie.delete("abc")
        self.assertEqual(count, 1)
        self.assertEqual(trie.size, 1)
        self.assertEqual(trie.unique, 1)

        # ensure ab remains intact
        count = trie.search("ab")
        self.assertEqual(count, 1)

        # ensure child c of ab was deleted upon removal of abc, but ab remain properly structured
        node = trie.root.get_child("a").get_child("b").get_child("c")
        self.assertIsNone(node)
        node = trie.root.get_child("a").get_child("b")
        self.assertIsInstance(node, TrieNode)
        self.assertTrue(node.empty())
        node = trie.root.get_child("a")
        self.assertIsInstance(node, TrieNode)

        # delete ab (last word in trie), ensure count is returned, ensure size, unique are updated
        count = trie.delete("ab")
        self.assertEqual(count, 1)
        self.assertEqual(trie.size, 0)
        self.assertEqual(trie.unique, 0)

        # ensure ab nodes were deleted but root remains intact
        self.assertIsInstance(trie.root, TrieNode)
        self.assertTrue(trie.root.empty())

        # verify deletions on realistic Trie with duplicates
        words = "on the banks of red cedar theres school thats known to all".split()

        # construct trie
        for word in words:
            trie.add(word)
        duplicates = words[:len(words) // 2]
        for word in duplicates:
            trie.add(word)
        anti_words = [word[::-1] for word in words]

        # delete from trie to ensure proper count is returned and size, unique are maintained
        size = len(words) + len(duplicates)
        unique = len(words)
        for word, anti_word in zip(words, anti_words):
            count = trie.delete(word)           # delete legitimate word
            if word in duplicates:              # check return and size
                self.assertEqual(count, 2)
                size -= 2
                self.assertEqual(trie.size, size)
            else:
                self.assertEqual(count, 1)
                size -= 1
                self.assertEqual(trie.size, size)

            # check unique for both cases of legitimate deletion
            unique -= 1
            self.assertEqual(trie.unique, unique)

            # ensure nonexistent words return zero and do not change size, unique
            count = trie.delete(anti_word)
            self.assertEqual(count, 0)
            self.assertEqual(trie.size, size)
            self.assertEqual(trie.unique, unique)

        # after all nodes are deleted, check to ensure trie is properly deconstructed
        self.assertIsInstance(trie.root, TrieNode)
        self.assertTrue(trie.root.empty())
        self.assertEqual(trie.size, 0)
        self.assertEqual(trie.unique, 0)

    def test_len_contains_empty(self):
        """
        Test len(Trie), in Trie and Trie.empty methods
        """
        # construct Trie and check empty
        trie = Trie()
        self.assertTrue(trie.empty())

        # build trie
        words = "on the banks of red cedar theres school thats known to all".split()
        for word in words:
            trie.add(word)

        # check len, in, empty operators
        self.assertEqual(len(trie), len(words))
        self.assertFalse(trie.empty())
        anti_words = [word[::-1] for word in words]
        for word, anti_word in zip(words, anti_words):
            self.assertTrue(word in trie)
            self.assertFalse(anti_word in trie)

        # add duplicates
        duplicates = words[:len(words) // 2]
        for word in duplicates:
            trie.add(word)

        # check len, in, empty operators again with duplicates
        self.assertEqual(len(trie), len(words) + len(duplicates))
        self.assertFalse(trie.empty())
        anti_words = [word[::-1] for word in words]
        for word, anti_word in zip(words, anti_words):
            self.assertTrue(word in trie)
            self.assertFalse(anti_word in trie)

    def test_get_vocabulary(self):
        """
        Test Trie.get_vocabulary method
        """
        # test empty Trie
        trie = Trie()
        vocab = trie.get_vocabulary()
        expected_vocab = {}
        self.assertEqual(vocab, expected_vocab)

        # build trie
        words = "on the banks of the red cedar theres school thats known to all".split()
        words.sort()
        for word in words:
            trie.add(word)

        # check entire vocabulary matches (alphabetically-sorted) words
        vocab = trie.get_vocabulary()
        expected_vocab.update([(word, 1) for word in words])
        expected_vocab["the"] = 2
        self.assertEqual(vocab, expected_vocab)

        # check prefixed vocabulary construction
        o_words = trie.get_vocabulary(prefix="o")
        self.assertEqual(o_words, {"of": 1, "on": 1})
        t_words = trie.get_vocabulary(prefix="t")
        self.assertEqual(t_words, {"thats": 1, "the": 2, "theres": 1, "to": 1})
        the_words = trie.get_vocabulary(prefix="the")
        self.assertEqual(the_words, {"the": 2, "theres": 1})

        # check empty prefix vocabulary construction
        expected_vocab = {}
        z_words = trie.get_vocabulary(prefix="z")
        self.assertEqual(z_words, expected_vocab)
        michigan_words = trie.get_vocabulary(prefix="michigan")
        self.assertEqual(michigan_words, expected_vocab)

    def test_spell_checker(self):
        """
        Test Trie.autocomplete method
        """
        # test empty Trie
        trie = Trie()
        vocab = trie.autocomplete("...")
        expected_vocab = {}
        self.assertEqual(vocab, expected_vocab)

        # build trie
        words = "on the banks of the red cedar theres school thats known to all".split()
        for word in words:
            trie.add(word)

        # test full-word search
        vocab = trie.autocomplete("the")
        self.assertEqual(vocab, {"the": 2})
        vocab = trie.autocomplete("on")
        self.assertEqual(vocab, {"on": 1})
        vocab = trie.autocomplete("cedar")
        self.assertEqual(vocab, {"cedar": 1})
        vocab = trie.autocomplete("notinvocab")
        self.assertEqual(vocab, {})

        # test partial-word search
        vocab = trie.autocomplete("b...s")
        self.assertEqual(vocab, {"banks": 1})
        vocab = trie.autocomplete("..d")
        self.assertEqual(vocab, {"red": 1})
        vocab = trie.autocomplete("....n")
        self.assertEqual(vocab, {"known": 1})
        vocab = trie.autocomplete("t.....")
        self.assertEqual(vocab, {"theres": 1})
        vocab = trie.autocomplete("t..")
        self.assertEqual(vocab, {"the": 2})
        vocab = trie.autocomplete("o.")
        self.assertEqual(vocab, {"on": 1, "of": 1})
        vocab = trie.autocomplete("..........")
        self.assertEqual(vocab, {})

        # test free-word search
        vocab = trie.autocomplete("..")
        self.assertEqual(vocab, {"on": 1, "of": 1, "to": 1})
        vocab = trie.autocomplete("...")
        self.assertEqual(vocab, {"the": 2, "red": 1, "all": 1})
        vocab = trie.autocomplete(".....")
        self.assertEqual(vocab, {"banks": 1, "cedar": 1, "thats": 1, "known": 1})
        vocab = trie.autocomplete("......")
        self.assertEqual(vocab, {"theres": 1, "school": 1})

    def test_comprehensive(self):
        """
        Test all Trie methods with large dataset
        """
        # construct trie from movie reviews text corpus
        trie = Trie()
        self.assertTrue(trie.empty())
        with open("movie_reviews_test.txt") as tsv:
            lines = tsv.readlines()
            for line in lines[1:]:                      # skip header row
                s = line.split("\t")[0]          # split tab-delimited variables
                s = "".join([c for c in s.lower() if c in string.ascii_lowercase or c.isspace()])  # clean
                for word in s.split():
                    trie.add(word)

        # check size, unique
        self.assertEqual(len(trie), 24040)
        self.assertEqual(trie.size, 24040)
        self.assertEqual(trie.unique, 5221)
        self.assertFalse(trie.empty())

        # search for various words
        count = trie.search("film")
        self.assertEqual(count, 179)
        self.assertTrue("film" in trie)
        count = trie.search("hacker")
        self.assertEqual(count, 2)
        self.assertTrue("hacker" in trie)
        count = trie.search("casino")
        self.assertEqual(count, 1)
        self.assertTrue("casino" in trie)
        count = trie.search("supercalifragilisticexpialidocious")
        self.assertEqual(count, 0)
        self.assertTrue("supercalifragilisticexpialidocious" not in trie)

        # delete words and ensure proper counts are returned
        count = trie.delete("episode")
        self.assertEqual(count, 5)
        self.assertTrue("episode" not in trie)
        count = trie.delete("the")
        self.assertEqual(count, 1415)
        self.assertTrue("the" not in trie)
        count = trie.delete("movie")
        self.assertEqual(count, 165)
        self.assertTrue("movie" not in trie)
        count = trie.delete("supercalifragilisticexpialidocious")
        self.assertEqual(count, 0)
        self.assertTrue("supercalifragilisticexpialidocious" not in trie)

        # check size, unique after deletions
        self.assertEqual(len(trie), 22455)
        self.assertEqual(trie.size, 22455)
        self.assertEqual(trie.unique, 5218)

        # check get_vocabulary with prefixes
        vocab = trie.get_vocabulary("watch")        # all words starting with watch
        expected_vocab = {'watch': 34, 'watchabilitybr': 1, 'watchable': 2, 'watched': 7, 'watching': 16}
        self.assertEqual(vocab, expected_vocab)
        vocab = trie.get_vocabulary("fun")          # all words starting with fun
        expected_vocab = {'fun': 4, 'funbr': 1, 'function': 1, 'funny': 17, 'funnybadbr': 1}
        self.assertEqual(vocab, expected_vocab)
        vocab = trie.get_vocabulary("z")            # all words starting with z
        expected_vocab = {'z': 1, 'zaitung': 1, 'zaniness': 1, 'zany': 1, 'zelah': 1, 'zenias': 2, 'zero': 3,
                          'zhang': 2, 'zimbalist': 1, 'ziyi': 2, 'zombie': 6, 'zombies': 6, 'zoom': 1}
        self.assertEqual(vocab, expected_vocab)

        # check autocomplete
        vocab = trie.autocomplete("...ed")         # all 5 letter words ending in ed
        expected_vocab = {'acted': 5, 'added': 2, 'aimed': 2, 'asked': 1, 'based': 6, 'bored': 5, 'dared': 1,
                          'dated': 3, 'ended': 2, 'faced': 1, 'fired': 1, 'hired': 1, 'holed': 1, 'jaded': 1,
                          'jared': 1, 'liked': 6, 'lived': 1, 'loved': 7, 'moved': 1, 'named': 4, 'paced': 2,
                          'rated': 3, 'tired': 3, 'toned': 1, 'tried': 5}
        self.assertEqual(vocab, expected_vocab)
        vocab = trie.autocomplete("..ing")         # all 5 letter words ending in ing
        expected_vocab = {'being': 28, 'bring': 4, 'doing': 7, 'dying': 2, 'going': 8,
                          'thing': 8, 'tying': 1, 'using': 3, 'vying': 1}
        self.assertEqual(vocab, expected_vocab)
        vocab = trie.autocomplete("th..")          # all 4 letter words starting with th
        expected_vocab = {'than': 50, 'that': 280, 'them': 29, 'then': 21, 'they': 75, 'this': 324, 'thus': 4}
        self.assertEqual(vocab, expected_vocab)

    def test_trie_classifier_artificial(self):
        """
        Test TrieClassifier.fit and TrieClassifier.predict on artificial datasets
        """
        # binary classifier: positive/negative-sentiment sentences
        classes = ["positive", "negative"]
        train_positive = ["sun sunny sunshine",
                          "smile smiling smiled",
                          "laugh laughing laughed",
                          "happy happier happiest"]
        train_negative = ["rain rainy rained",
                          "frown frowning frowned",
                          "cry crying cried",
                          "sad sadder saddest"]

        training_strings = {"positive": train_positive, "negative": train_negative}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)

        # test single predictions
        pred = clf.predict(["the sunshine made me smile today"])
        self.assertEqual(pred, ["positive"])
        pred = clf.predict(["laughing with my best friends always makes me happier"])
        self.assertEqual(pred, ["positive"])
        pred = clf.predict(["the clouds and rain always make me sad"])
        self.assertEqual(pred, ["negative"])
        pred = clf.predict(["she frowned and cried after hearing the bad news"])
        self.assertEqual(pred, ["negative"])

        # test multiple predictions
        test_positive = ["the sunshine made me smile today",
                         "laughing with my best friends always makes me happier",
                         "when youre happy you dont frown or cry"]
        pred = clf.predict(test_positive)
        self.assertEqual(pred, ["positive", "positive", "negative"])
        truth = ["positive" for _ in test_positive]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 2/3)

        test_negative = ["the clouds and rain always make me sad",
                         "she had not laughed nor smiled for days",
                         "without sunshine she found it hard to be happy"]
        pred = clf.predict(test_negative)
        self.assertEqual(pred, ["negative", "positive", "positive"])
        truth = ["negative" for _ in test_positive]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 1/3)

        # multi-classifier: subject-area strings
        classes = ["computer", "business", "history", "art"]
        train_computer = ["computer computers computed computing computational",
                          "program programs programmed programming programmer",
                          "code codes coded coding coder"]
        train_business = ["present presents presented presenting presentation",
                          "negotiate negotiates negotiated negotiating negotiation",
                          "sell sells sold selling"]
        train_history = ["research researches researched researching researcher",
                         "write writes wrote writing",
                         "read reads reading"]
        train_art = ["paint paints painted painting",
                     "draw draws drew drawing",
                     "sculpt sculpts sculpted sculpting sculpture"]
        training_strings = {"computer": train_computer, "business": train_business,
                              "history": train_history, "art": train_art}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)

        test_all = ["she enjoys writing computer programs",
                    "the ceo presented an amazing product and sold the team on her idea",
                    "she aspires to be a history professor who writes research papers on ancient greece",
                    "she drew a sketch of the sculpture she envisioned before sculpting it",
                    "as a computer science researcher she reads and writes code all day",
                    "she negotiated a deal with the museum to display her latest masterpiece at the new exhibit",
                    "the business college established a new program in supply chain last summer"]
        pred = clf.predict(test_all)
        self.assertEqual(pred, ["computer", "business", "history", "art", "history", "business", "computer"])
        truth = ["computer", "business", "history", "art", "computer", "art", "business"]
        acc = clf.accuracy(truth, pred)
        self.assertAlmostEqual(acc, 4/7)

    def test_trie_classifier_real(self):
        """
        Test TrieClassifier.fit and TrieClassifier.predict on real dataset
        """
        # load training data
        classes = ["positive", "negative"]
        train_positive, train_negative = [], []
        with open("movie_reviews_train.txt") as tsv:
            lines = tsv.readlines()
            for line in lines[1:]:                          # skip header row
                sentence, sentiment = line.split("\t")      # split tab-delimited variables
                sentence = "".join([c for c in sentence.lower()
                                    if c in string.ascii_lowercase or c.isspace()])     # clean sentence and sentiment
                sentiment = sentiment.strip()
                if sentiment == "positive":
                    train_positive.append(sentence)
                else:
                    train_negative.append(sentence)

        # fit classifier
        training_strings = {"positive": train_positive, "negative": train_negative}
        clf = TrieClassifier(classes)
        clf.fit(training_strings)

        # load testing data
        test_strings, truth = [], []
        with open("movie_reviews_test.txt") as tsv:
            lines = tsv.readlines()
            for line in lines[1:]:                      # skip header row
                sentence, sentiment = line.split("\t")  # split tab-delimited variables
                sentence = "".join([c for c in sentence.lower()
                                    if c in string.ascii_lowercase or c.isspace()])     # clean sentence and sentiment
                sentiment = sentiment.strip()
                test_strings.append(sentence)
                truth.append(sentiment)

        # predict and validate
        pred = clf.predict(test_strings)
        expected_pred = ['negative', 'negative', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive',
                         'positive', 'negative', 'negative', 'negative', 'negative', 'negative', 'positive', 'positive',
                         'positive', 'negative', 'negative', 'negative', 'negative', 'positive', 'positive', 'positive',
                         'positive', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'positive',
                         'negative', 'negative', 'negative', 'negative', 'negative', 'positive', 'negative', 'negative',
                         'negative', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative', 'positive',
                         'negative', 'negative', 'positive', 'positive', 'positive', 'negative', 'negative', 'negative',
                         'positive', 'negative', 'positive', 'negative', 'negative', 'negative', 'negative', 'negative',
                         'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative',
                         'negative', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'negative',
                         'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'negative', 'positive',
                         'negative', 'positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'negative',
                         'negative', 'negative', 'positive', 'negative']
        self.assertEqual(pred, expected_pred)
        acc = clf.accuracy(truth, pred)
        expected_acc = 0.61
        self.assertAlmostEqual(acc, expected_acc)


if __name__ == '__main__':
    unittest.main()
