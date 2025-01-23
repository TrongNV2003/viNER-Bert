import re
from typing import List

from underthesea import word_tokenize


class TextPreprocess:
    def __init__(self):
        pass

    def process_text(self, text: List[str]) -> str:
        """
        This function will process the text by removing stopwords, punctuation, and whitespace
        """
        
        string = []
        cleaned_text = self._remove_whitespace(text)
        cleaned_text = word_tokenize(cleaned_text)
        for word in cleaned_text:
            word = word.replace(" ", "_")
            string.append(word)
        return string

    def _remove_whitespace(self, text: str) -> str:
        """
        This function will remove whitespace from the text

        Parameters:
            text: str

        Returns:
            text without whitespace
        """

        cleaned_text = re.sub(r"\s+", " ", text).strip()
        return cleaned_text