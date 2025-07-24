import sys
from csv import reader

# lists of predefined words
FILLERS = ["um", "uh", "eh"]
BACKCHANNELS = ["hm", "yeah", "mhm", "huh"]
EXCEPTION_RULES = [
    "that 's that",
    "that is that",
    "as well as",
    "as much as",
    "as ADJ as",
    "the NOUN the",
    "do n't do",
]
PUNCT = [".", "?", "'"]


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


#
# read transcripts and return texts of a speaker only
def read_transcript(path, speaker_label=None):
    kit_col = 0
    wav_col = 1
    beg_col = 2
    end_col = 3
    txt_col = 4
    spkr_col = 5

    usable = False
    with open(path, newline="") as f:
        csv_reader = reader(f, delimiter="\t")
        first = tuple(next(csv_reader))

        if first == ("Kit", "Audio", "Beg", "End", "Text"):
            usable = True if speaker_label is None else False

        if first in (
            ("Kit", "Audio", "Beg", "End", "Text", "Speaker"),
            ("Kit", "Audio", "Beg", "End", "Text", "Speaker", "Section"),
        ):
            usable = True

        # no header, make a guess
        if is_float(first[beg_col]) and is_float(first[end_col]) and len(first) >= 5:
            second = tuple(next(csv_reader))
            if (
                first[kit_col] == second[kit_col]
                and first[wav_col] == second[wav_col]
                and is_float(second[beg_col])
                and is_float(second[beg_col])
            ):
                # we can use files without speaker labels IIF we don't specify a speaker_label
                # NB: this assumes that the file is just a single speaker's turns
                usable = True if (len(first) > 5 or speaker_label is None) else False

        if usable:
            f.seek(0)
            if speaker_label is not None:
                return " ".join(
                    row[txt_col] for row in csv_reader if row[spkr_col] == speaker_label
                )
            else:
                return " ".join(row[txt_col] for row in csv_reader)

        else:
            return ""


# update the list of previous words to count repetitions
def update_words(prev_bi, prev, word):
    penult_bi = prev_bi
    prev_bi = prev + word.text.lower()
    penult = prev
    prev = word.text.lower()
    prev_pos = word.pos_
    return penult_bi, prev_bi, penult, prev, prev_pos


# count fillers (uses previously defined list)
def count_fillers(word, um, uh, eh):
    if word in FILLERS:
        if word == "um":
            um += 1
        elif word == "uh":
            uh += 1
        else:
            eh += 1
    else:
        pass


# count backchannels (uses previously defined list)
def count_backchannels(word, hm, yeah):
    if word in BACKCHANNELS:
        if word == "hm":
            hm += 1
        elif word == "yeah":
            yeah += 1
        else:
            pass
    else:
        pass


# clean the transcripts by excluding dysfluency markers and repetitions => This helps improving the pos tagging accuracy.
def clean_doc(doc):
    # initiate dysfluency marker counts
    um = uh = eh = hm = yeah = partial = restart = 0
    repetitionList = []

    # initiate previous words
    prev_bi = penult_bi = prev = penult = prev_pos = "NA"

    # list of cleaned words to be used for second-pass pos tagging
    cleaned = []

    # loope through words in a doc
    for word in doc:
        if word.pos_ != "SPACE" and word.text != ",":
            # count dysfluency markers
            if (
                (word.text.lower() in FILLERS)
                or (word.text.lower() in BACKCHANNELS)
                or word.text.endswith("-")
                or word.text.endswith("=")
                or (word.text == "#")
            ):
                count_fillers(word.text.lower(), um, uh, eh)
                count_backchannels(word.text.lower(), hm, yeah)
                if word.text.endswith("-"):
                    partial += 1
                elif word.text.endswith("="):
                    repetitionList.append(word.text)
                elif word.text == "#":
                    restart += 1

            else:
                # remove repetitions
                if (
                    word.text.lower() != prev
                ):  # check if a given word is a repetition of a previous word (e.g., this, <this> boy is ...)
                    if (
                        word.text.lower() != penult
                    ):  # check if a given word is a repetition of a preceding word of the previous word (e.g., this, uh, <this> boy ... )
                        # if a phrase is not repeated, do not count as a repetition (e.g., This boy, this girl is ... )
                        if (
                            prev_bi != prev.lower() + word.text.lower()
                            and penult_bi != prev.lower() + word.text.lower()
                        ):
                            cleaned.append(word.text)
                            penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                                prev_bi, prev, word
                            )
                        else:
                            cleaned.pop(-1)
                    # do not count as repetition if a phrase is in the exception rules (e.g., as soon <as> possible)
                    elif (
                        penult + " " + prev + " " + word.text.lower() in EXCEPTION_RULES
                        or penult + " " + prev_pos + " " + word.text.lower()
                        in EXCEPTION_RULES
                    ):
                        cleaned.append(word.text)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                            prev_bi, prev, word
                        )
                    # if the preceding word is a sentence boundary, do not count as a repetition (e.g., She ate this. <This> boy is ...)
                    elif word.text.lower() == penult and (prev == "." or prev == "?"):
                        cleaned.append(word.text)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                            prev_bi, prev, word
                        )
                    # pass repetition of punctuation marks
                    elif word.text in PUNCT and penult in PUNCT:
                        pass
                    # if a given word is a sentence boundary, do not count as a repetition
                    elif word.text == "." or word.text == "?":
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                            prev_bi, prev, word
                        )

                    else:  # if a preceding word of a previous word is a copy of the given word, count it as a repetition
                        repetitionList.append(word)
                        penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                            prev_bi, prev, word
                        )

                elif word.text in PUNCT and prev in PUNCT:
                    pass

                elif word.text == "mm" and prev == "mm":
                    cleaned.pop(-1)
                elif word.text == "mhm" and prev == "mhm":
                    cleaned.pop(-1)

                else:  # if a previous word is an exact copy of the given word, count it as a repetition
                    repetitionList.append(word)
                    penult_bi, prev_bi, penult, prev, prev_pos = update_words(
                        prev_bi, prev, word
                    )

    return (
        " ".join(cleaned),
        len(cleaned),
        um,
        uh,
        eh,
        hm,
        yeah,
        partial,
        len(repetitionList),
        restart,
    )
