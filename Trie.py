"""
Author - Jeffrey Valentic
Tries
"""

from __future__ import annotations
from typing import Tuple, Dict, List


class TrieNode:
    """
    Implementation of a trie node.
    """

    # DO NOT MODIFY

    __slots__ = "children", "is_end"

    def __init__(self, arr_size: int = 26) -> None:
        """
        Constructs a TrieNode with arr_size slots for child nodes.
        :param arr_size: Number of slots to allocate for child nodes.
        :return: None
        """
        self.children = [None] * arr_size
        self.is_end = 0

    def __str__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        if self.empty():
            return "..."
        children = self.children  # to shorten proceeding line
        return str({chr(i + ord("a")) + "*"*min(children[i].is_end, 1): children[i] for i in range(26) if children[i]})

    def __repr__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        return self.__str__()

    def __eq__(self, other: TrieNode) -> bool:
        """
        Compares two TrieNodes for equality.
        :return: True if two TrieNodes are equal, else False
        """
        if not other or self.is_end != other.is_end:
            return False
        return self.children == other.children

    # Implement Below

    def empty(self) -> bool:
        """
        Checks if the node has any populated children slots.
        :return: Returns true if there are no children.
        """
        for child in self.children:
            if child is not None:
                return False
        return True

    @staticmethod
    def _get_index(char: str) -> int:
        """
        Provides the index value of a char (0-25)
        :param char: letter to check
        :return: int 0-25 representing the location of that letter
        """
        if char.isupper():
            char = char.lower()
        return ord(char) - 97

    def get_child(self, char: str) -> TrieNode:
        """
        Gets a child that matches the char passed in
        :param char: char to find
        :return: returns the node at char
        """
        char_ind = self._get_index(char)
        return self.children[char_ind]

    def set_child(self, char: str) -> None:
        """
        Sets the child of a node to the arg char.
        :param char: value to set as child
        :return: none
        """
        char_ind = self._get_index(char)
        self.children[char_ind] = TrieNode()

    def delete_child(self, char: str) -> None:
        """
        Removes child node from trie
        :param char: node char to remove
        :return: none
        """
        char_ind = self._get_index(char)
        self.children[char_ind] = None


class Trie:
    """
    Implementation of a trie.
    """

    __slots__ = "root", "unique", "size"

    def __init__(self) -> None:
        """
        Constructs an empty Trie.
        :return: None.
        """
        self.root = TrieNode()
        self.unique = 0
        self.size = 0

    def __str__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return "Trie Visual:\n" + str(self.root)

    def __repr__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return self.__str__()

    def __eq__(self, other: Trie) -> bool:
        """
        Compares two Tries for equality.
        :return: True if two Tries are equal, else False
        """
        return self.root == other.root

    def add(self, word: str) -> int:
        """
        Adds a new word to the trie
        :param word: word to add
        :return: returns the is_end of the word added
        """
        def add_inner(node: TrieNode, index: int) -> int:

            if index < len(word):
                if node.get_child(word[index]) is None:
                    node.set_child(word[index])

                return add_inner(node.get_child(word[index]), index+1)

            else:
                if node.is_end == 0:
                    self.unique += 1
                    node.is_end += 1
                else:
                    node.is_end += 1
                return node.is_end

        self.size += 1
        result = add_inner(self.root, 0)
        return result

    def search(self, word: str) -> int:
        """
        Determines if a word is in a trie
        :param word: word to look for
        :return: is_end of the word
        """

        def search_inner(node: TrieNode, index: int) -> int:

            if index < len(word):

                child = node.get_child(word[index])

                if child is not None:
                    return search_inner(child, index+1)

                return 0

            return node.is_end

        return search_inner(self.root, 0)

    def delete(self, word: str) -> int:
        """
        Removes a word from a trie
        :param word: word to remove
        :return: returns the is_end of the word removed.
        """
        def delete_inner(node: TrieNode, index: int) -> Tuple[int, bool]:

            count = 0
            child = node.get_child(word[index])
            child_ind = node._get_index(word[index])

            if len(word) == index + 1:
                count = child.is_end
                self.size -= count

                if child.is_end > 0:
                    child.is_end = 0

                    if child.empty():
                        node.delete_child(word[index])
                        return count, True

                    return count, False

                return count, False

            else:
                count, remove_child = delete_inner(node.children[child_ind], index+1)

                if remove_child:

                    if child.is_end == 0:
                        return count, True

                    return count, False

                return count, False

        if self.search(word) > 0:
            word_count, remove_cur = delete_inner(self.root, 0)
            self.unique -= 1

            if self.size == 0:
                blank_trie = Trie()
                self.root = blank_trie.root

            return word_count

        return 0

    def __len__(self) -> int:
        """
        length of the trie
        :return: .size of trie
        """
        return self.size

    def __contains__(self, word: str) -> bool:
        """
        determines if a word is in a trie
        :param word: word to look for
        :return: True if word is found, else false.
        """
        if self.search(word) > 0:
            return True
        return False

    def empty(self) -> bool:
        """
        Checks if trie is empty
        :return: True if empty, else false.
        """
        if self.size == 0:
            return True
        return False

    def get_vocabulary(self, prefix: str = "") -> Dict[str, int]:
        """
        gets any words in trie that begin with prefix
        :param prefix: start of words to look for
        :return: Returns dict of word and is_end of word.
        """

        vocab = {}

        def get_char(char_ind):
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            return alphabet[char_ind]

        def get_vocabulary_inner(node, suffix):
            child_ind = -1

            for child in node.children:
                child_ind += 1

                if child is not None and child.is_end != 0:
                    old_suf = suffix
                    suffix = suffix + get_char(child_ind)
                    key = prefix + suffix
                    vocab[key] = child.is_end
                    get_vocabulary_inner(child, suffix)
                    suffix = old_suf

                elif child is not None:
                    old_suf = suffix
                    suffix = suffix + get_char(child_ind)
                    get_vocabulary_inner(child, suffix)
                    suffix = old_suf

        start_node = self.root
        if prefix == '':
            get_vocabulary_inner(start_node, "")
            return vocab

        if start_node.children[TrieNode._get_index(prefix[0])] is None:
            return vocab

        else:
            for char in prefix:
                if start_node.children[TrieNode._get_index(char)] is None:
                    get_vocabulary_inner(start_node, "")
                    return vocab
                else:
                    start_node = start_node.children[TrieNode._get_index(char)]
            if start_node.is_end != 0:
                vocab[prefix] = start_node.is_end
            get_vocabulary_inner(start_node, "")
            return vocab

    def autocomplete(self, word: str) -> Dict[str, int]:
        """
        Provides all words that match the characters in word and length of word.
        :param word: the template for what to compare to.
        :return: dict of all words that follow the template word and their is_end.
        """

        predictions = {}

        def autocomplete_inner(node, prefix, index):
            if index < len(word):
                if word[index] == '.':
                    for c_ind in range(len(node.children)):
                        if node.children[c_ind] is not None:
                            new_prefix = prefix + chr(c_ind+97)
                            new_node = node.get_child(chr(c_ind+97))
                            autocomplete_inner(new_node, new_prefix, index+1)

                else:
                    char_ind = TrieNode._get_index(word[index])
                    if node.children[char_ind] is not None:
                        new_prefix = prefix + word[index]
                        new_node = node.get_child(chr(char_ind + 97))
                        autocomplete_inner(new_node, new_prefix, index + 1)

            elif node.is_end > 0:
                predictions[prefix] = node.is_end
            else:
                return

        autocomplete_inner(self.root, '', 0)

        return predictions


class TrieClassifier:
    """
    Implementation of a trie-based text classifier.
    """

    __slots__ = "tries"

    def __init__(self, classes: List[str]) -> None:
        """
        Constructs a TrieClassifier with specified classes.
        :param classes: List of possible class labels of training and testing data.
        :return: None.
        """
        self.tries = {}
        for cls in classes:
            self.tries[cls] = Trie()

    @staticmethod
    def accuracy(labels: List[str], predictions: List[str]) -> float:
        """
        Computes the proportion of predictions that match labels.
        :param labels: List of strings corresponding to correct class labels.
        :param predictions: List of strings corresponding to predicted class labels.
        :return: Float proportion of correct labels.
        """
        correct = sum([1 if label == prediction else 0 for label, prediction in zip(labels, predictions)])
        return correct / len(labels)

    def fit(self, class_strings: Dict[str, List[str]]) -> None:
        """
        Adds words to trie dict based on their class (dict keys)
        :param class_strings: dict with key = classification and val = list of words to add.
        :return: none
        """
        for key, val in class_strings.items():
            for text in val:
                word_lst = text.split()
                for word in word_lst:
                    self.tries[key].add(word)

        return

    def predict(self, strings: List[str]) -> List[str]:
        """
        Determines what class a tweet belongs to.
        :param strings: list of tweets to classify
        :return: list of classifications
        """
        prediction_lst = []

        for tweet in strings:
            class_scores = ('', -1)
            word_lst = tweet.split()
            for cls in self.tries.keys():
                cur_score = 0
                for word in word_lst:
                    cur_score += self.tries[cls].search(word)

                cur_score = cur_score / len(self.tries[cls])
                if cur_score > class_scores[1]:
                    class_scores = (cls, cur_score)

            prediction_lst.append(class_scores[0])

        return prediction_lst