### Usage1:
# python3 lexical_pipeline.py -output_file ../speechbiomarkers/summarized_lexical_20210413.csv -input_folder ../speechbiomarkers/picture -filetype .txt
## V0.3.2: 2023-April-4

## Check if all packages are installed and if not, install them.


import numpy as np
import pandas as pd
import spacy
from lexicalrichness import LexicalRichness

from lexicalLibs.count_words import convert100, count_pos
from lexicalLibs.prep_text import clean_doc, read_transcript
from lexicalLibs.rate_words import attach_lexical, get_phondict, lexical_summary

# location of the aggregated lexical measure file


# load nlp model
nlp = spacy.load("en_core_web_lg")


class Predictor:
    def __init__(self, measureDict: pd.DataFrame, full_df: bool):
        self.measureDict = measureDict
        self.phonDf = get_phondict()
        if full_df:
            self.full_df = pd.DataFrame()
        else:
            self.full_df = None

    def predict(
        self,
        text: str,
        filename: str | None = None,
    ) -> pd.DataFrame:
        # run first-pass pos tagging on the entire doc (POS tags are used to exclude repetitions)
        doc = nlp(text)
        # clean the doc
        cleaned_text, total_word, um, uh, eh, hm, yeah, partial, repetition, restart = (
            clean_doc(doc)
        )
        # tally the total word count + dysfluency markers
        total = total_word + um + uh + eh + hm + yeah + partial + repetition
        # run second-pass pos tagging on cleaned texts only (without dysfluency markers)
        doc = nlp(cleaned_text)
        # get lexical diversity measures (lex is used to calculate lexical diversity)
        lex = LexicalRichness(cleaned_text)

        # count pos categories and dysfluency markers
        (
            pos_counts,
            tag_counts,
            uniqueAllCount,
            uniqueContentCount,
            uniqueNounCount,
            uniqueAdjCount,
            uniqueVerbCount,
            uniqueAdvCount,
        ) = count_pos(doc)
        other_dict = {
            "um": um,
            "uh": uh,
            "eh": eh,
            "hm": hm,
            "yeah": yeah,
            "partial": partial,
            "repetition": repetition,
            "restart": restart,
            "uniqueAll": uniqueAllCount,
            "uniqueContent": uniqueContentCount,
            "uniqueNoun": uniqueNounCount,
            "uniqueAdj": uniqueAdjCount,
            "uniqueVerb": uniqueVerbCount,
            "uniqueAdv": uniqueAdvCount,
        }

        # combine the two dicts
        pos_counts = {**pos_counts, **other_dict}
        # convert to counts per 100 words
        pos100 = convert100(dict(pos_counts), total)
        tag100 = convert100(dict(tag_counts), total)
        # make the counts as a data frame
        pos_all = pd.DataFrame({**pos100, **tag100}, index=[0])

        # rate lexical measures
        lexical = attach_lexical(doc, self.measureDict, self.phonDf)
        if self.full_df is not None and filename is not None:
            lexical["filename"] = filename
            # combine pos and dysfluency counts and lexical measures
            self.full_df = pd.concat([self.full_df, lexical])
            lexical = lexical.drop(["filename"], axis=1)

        # calculate averaged lexical measure
        lexicalSumDF = lexical_summary(lexical)
        # combine counts and lexical measure dfs
        result = pd.concat([pos_all, lexicalSumDF], axis=1)
        # add additional columns to the combined df
        result["total_words"] = total_word
        result["total_words_plus_others"] = total

        # calculate lexical diversity by window and document (lex.words) size
        if (
            lex.words > 25
        ):  # if the doc contains more than 25 words, calculate lexical diversity for all window size
            result["lexical_diversity_25"] = lex.mattr(window_size=25)
            result["lexical_diversity_20"] = lex.mattr(window_size=20)
            result["lexical_diversity_15"] = lex.mattr(window_size=15)
        elif lex.words > 20:
            result["lexical_diversity_25"] = np.nan
            result["lexical_diversity_20"] = lex.mattr(window_size=20)
            result["lexical_diversity_15"] = lex.mattr(window_size=15)
        elif lex.words > 15:
            result["lexical_diversity_25"] = np.nan
            result["lexical_diversity_20"] = np.nan
            result["lexical_diversity_15"] = lex.mattr(window_size=15)
        else:  # if doc contains fewer than 15 words, lexical diversity is not calculated
            result["lexical_diversity_25"] = np.nan
            result["lexical_diversity_20"] = np.nan
            result["lexical_diversity_15"] = np.nan
        return result


if __name__ == "__main__":
    import argparse
    from os import listdir
    from os.path import dirname, isdir
    from os.path import join as join_path

    def get_lexical_measures():
        return pd.read_csv(join_path(dirname(__file__), "all_measures_raw.csv"))

    def get_file_list(path: str, suffix: str) -> list[str]:
        ret: list[str] = []
        if isdir(path):
            for f in listdir(path):
                if f.endswith(suffix):
                    ret.append(f)
        return ret

    # main function
    def main(args):
        # define output file
        outputname = args.output_file

        # get a list of files to process
        filelist = get_file_list(args.input_folder, args.filetype)

        print(
            "List of files to be processed: \n",
            "\n".join(filelist),
            "\n If empty, check your directory path and file extension again.",
        )
        if not filelist:
            return None

        # initiate result dataframes
        allResults = pd.DataFrame()

        predictor = Predictor(get_lexical_measures(), True)

        # loop through the file list
        for fname in filelist:
            file = join_path(args.input_folder, fname)
            print(file, " is being processed...")

            # read a transcript (transcripts need to be tab-separated!)
            text = read_transcript(file, args.speaker_label)

            result = predictor.predict(text, fname)
            result["filename"] = fname
            #
            # update the allResults df with the processed doc
            allResults = pd.concat([allResults, result], sort=True)
        # define column names for all pos categories
        col_na = [
            "ADJ",
            "ADP",
            "ADV",
            "CC",
            "CCONJ",
            "CD",
            "DET",
            "DT",
            "EX",
            "FW",
            "IN",
            "INTJ",
            "JJ",
            "JJR",
            "JJS",
            "MD",
            "NN",
            "NNP",
            "NNPS",
            "NNS",
            "NOUN",
            "NUM",
            "PART",
            "PDT",
            "POS",
            "PRON",
            "PROPN",
            "PRP",
            "PRP$",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "TO",
            "UH",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "VERB",
            "WDT",
            "WP",
            "WP$",
            "WRB",
            "X",
            "XX",
        ]
        # if the doc included zero instances of a given pos count, insert 0 for zero count
        for col in col_na:
            if col in allResults:
                allResults[col].fillna(0, inplace=True)
            else:
                allResults[col] = 0

        # count the number of tense_inflected verbs
        allResults["tense_inflected_verb"] = (
            allResults["MD"] + allResults["VBD"] + allResults["VBP"] + allResults["VBZ"]
        )
        # count total filler counts
        allResults["filler"] = allResults["um"] + allResults["uh"] + allResults["eh"]
        # output allResults df (this is a word by word dataframe)
        allResults.to_csv(outputname, index=False)
        # output a summarized simple result file (this is for collaborators)
        smalldf = allResults[
            [
                "filename",
                "NOUN",
                "VERB",
                "ADJ",
                "ADV",
                "ADP",
                "DET",
                "PRON",
                "CCONJ",
                "PART",
                "NUM",
                "filler",
                "partial",
                "repetition",
                "tense_inflected_verb",
                "lexical_diversity_15",
                "total_words",
                "total_words_plus_others",
                "uniqueContent",
                "concreteness_content",
                "frequency_content",
                "AoA_content",
                "familiarity_content",
                "phone_content",
                "ambiguity_content",
                "total_syll",
            ]
        ]
        external_filename = outputname.split(".")[0] + "_simple.csv"
        smalldf.to_csv(external_filename, index=False)
        # output a summarized full result file (this is generally for internal use.)
        full_filename = outputname.split(".")[0] + "_full.csv"

        predictor.full_df.to_csv(full_filename, index=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-output_file", type=str, required=True, help="Name the output file"
    )
    parser.add_argument(
        "-input_folder",
        type=str,
        required=True,
        help="Folder containing input documents",
    )
    parser.add_argument(
        "-filetype", type=str, required=True, help="Input file extensions"
    )
    parser.add_argument(
        "-speaker_label", type=str, required=False, help="Speaker label of interest"
    )
    args = parser.parse_args()
    main(args)
