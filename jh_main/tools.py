def calculate_fres(
        pfx_text: Annotated[str, "A patient-friendly explanation string."]
    ) -> dict:
        """Calculate the Flesch Reading Ease Score and estimated reading level for a given explanation."""
        
        def count_syllables(word):
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            if not word:
                return 0
            syllables = re.findall(r'[aeiouy]+', word)
            if word.endswith("e") and not word.endswith("le"):
                syllables = syllables[:-1]
            return max(1, len(syllables))

        sentences = re.split(r'[.!?]', pfx_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)

        words = re.findall(r'\b\w+\b', pfx_text)
        num_words = len(words)
        num_syllables = sum(count_syllables(word) for word in words)

        if num_sentences == 0 or num_words == 0:
            return {"error": "Input must contain at least one sentence and one word."}

        fres = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

        if fres >= 90:
            grade_level = "5th grade"
        elif fres >= 80:
            grade_level = "6th grade"
        elif fres >= 70:
            grade_level = "7th grade"
        elif fres >= 60:
            grade_level = "8th–9th grade"
        elif fres >= 50:
            grade_level = "10th–12th grade"
        elif fres >= 30:
            grade_level = "College"
        elif fres >= 10:
            grade_level = "College graduate"
        else:
            grade_level = "Professional"

        return {
            "FRES": round(fres, 2),
            "Reading_Level": grade_level
        }
