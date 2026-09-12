import re
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """
    Generic query expansion result.
    """

    original: str
    expanded: str
    keywords: list[str]
    years: list[str]
    numbers: list[str]
    entities: list[str]
    units: list[str]


class QueryExpander:
    """
    Simple, domain-independent query expansion.

    This class does not contain knowledge about:
    - specific companies
    - industries
    - products
    - documents
    - financial terminology

    It only extracts useful information from the user's question.
    """

    STOP_WORDS = {
        "what",
        "was",
        "were",
        "is",
        "are",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "for",
        "to",
        "from",
        "how",
        "many",
        "much",
        "does",
        "did",
        "do",
        "and",
        "or",
        "with",
        "about",
        "which",
        "where",
        "when",
        "who",
        "why",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "me",
        "give",
        "show",
        "according",
        "document",
        "documents",
        "report",
    }

    COMMON_UNITS = {
        "msek",
        "sek",
        "usd",
        "eur",
        "gbp",
        "inr",
        "million",
        "millions",
        "billion",
        "billions",
        "percent",
        "percentage",
        "%",
        "kg",
        "g",
        "mg",
        "km",
        "m",
        "cm",
        "mm",
        "mb",
        "gb",
        "tb",
        "kw",
        "mw",
        "ghz",
        "mhz",
        "hz",
        "psi",
        "bar",
        "rpm",
        "voltage",
        "volt",
        "v",
        "amp",
        "ampere",
        "a",
    }

    def _tokenize(self, text: str) -> list[str]:
        """
        Extract words and numbers from text.
        """

        return re.findall(
            r"\b[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*\b",
            text.lower(),
        )

    def _extract_years(self, question: str) -> list[str]:
        """
        Extract explicitly mentioned four-digit years.
        """

        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            question,
        )

        return list(dict.fromkeys(years))

    def _extract_numbers(self, question: str) -> list[str]:
        """
        Extract explicitly mentioned numeric values.
        """

        numbers = re.findall(
            r"\b\d+(?:[.,]\d+)?\b",
            question,
        )

        return list(dict.fromkeys(numbers))

    def _extract_units(self, question: str) -> list[str]:
        """
        Extract explicitly mentioned units.
        """

        tokens = self._tokenize(question)

        return list(
            dict.fromkeys(
                token
                for token in tokens
                if token in self.COMMON_UNITS
            )
        )

    def _extract_keywords(self, question: str) -> list[str]:
        """
        Extract meaningful words from the question.
        """

        tokens = self._tokenize(question)
        years = set(self._extract_years(question))

        keywords = []

        for token in tokens:
            if token in self.STOP_WORDS:
                continue

            if token in years:
                continue

            if token in self.COMMON_UNITS:
                continue

            if len(token) <= 1:
                continue

            keywords.append(token)

        return list(dict.fromkeys(keywords))

    def _extract_entities(self, question: str) -> list[str]:
        """
        Preserve likely named entities from the original question.

        This is intentionally lightweight and does not use
        a predefined company or product list.
        """

        pattern = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9&.-]*)(?:\s+[A-Z][A-Za-z0-9&.-]*)*\b"
        )

        matches = pattern.findall(question)

        entities = []

        for match in matches:
            normalized = match.strip()

            if not normalized:
                continue

            if normalized.lower() in self.STOP_WORDS:
                continue

            entities.append(normalized)

        return list(dict.fromkeys(entities))

    def expand(self, question: str) -> ExpandedQuery:
        """
        Create a simple retrieval-oriented query.

        The original question is always preserved.
        Only information already present in the question
        is added.
        """

        question = question.strip()

        if not question:
            return ExpandedQuery(
                original="",
                expanded="",
                keywords=[],
                years=[],
                numbers=[],
                entities=[],
                units=[],
            )

        years = self._extract_years(question)
        numbers = self._extract_numbers(question)
        units = self._extract_units(question)
        keywords = self._extract_keywords(question)
        entities = self._extract_entities(question)

        # Start with the original question.
        retrieval_terms = [question]

        # Add only keywords that are not already represented
        # in the original question as a complete phrase.
        original_lower = question.lower()

        for keyword in keywords:
            if keyword not in original_lower:
                retrieval_terms.append(keyword)

        # Explicit years/numbers/units are useful for retrieval,
        # especially for factual questions.
        for value in years + numbers + units:
            if value.lower() not in original_lower:
                retrieval_terms.append(value)

        # Remove duplicates while preserving order.
        unique_terms = []
        seen = set()

        for term in retrieval_terms:
            normalized = term.lower().strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_terms.append(term)

        expanded_query = " ".join(unique_terms)

        return ExpandedQuery(
            original=question,
            expanded=expanded_query,
            keywords=keywords,
            years=years,
            numbers=numbers,
            entities=entities,
            units=units,
        )